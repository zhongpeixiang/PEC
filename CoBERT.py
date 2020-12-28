import argparse
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import re
import string
import sys
import time
import json
import math
import copy
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

from util import load_pickle, save_pickle, count_parameters, compute_metrics, compute_metrics_from_logits
import logging

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")


def cprint(*args):
    text = ""
    for arg in args:
        text += "{0} ".format(arg)
    logging.info(text)

def tokenize_conversations(data, tokenizer, max_sent_len):
    new_data = []
    for conv in tqdm(data):
        new_conv = []
        for i, (speaker, sents) in enumerate(conv):
            # each utterance has been segmented into multiple sentences
            if i==0:
                word_limit = 90
            else:
                word_limit = max_sent_len
            
            tokenized_sent = []
            for sent in sents:
                tokenized = tokenizer.tokenize(sent)
                if len(tokenized_sent) + len(tokenized) <= word_limit:
                    tokenized_sent.extend(tokenized)
                else:
                    break
            if len(tokenized_sent) == 0:
                tokenized_sent = tokenized[:word_limit]
            new_conv.append((speaker, tokenized_sent))
        new_data.append(new_conv)
    return new_data

def tokenize_personas(data, tokenizer, all_speakers, num_personas):
    # average: each speaker corresponds to a list of tokens, separated by [SEP] between sents
    # memnet: each speaker corresponds to a 2D list of tokens 
    new_data = {}
    for k, sents in tqdm(data.items()):
        if k in all_speakers:
            tokenized_words = []
            for sent in sents[:num_personas]:
                tokenized_words.extend(tokenizer.tokenize(" ".join(sent))[:22] + ["[SEP]"])
            if len(tokenized_words) > 1:
                tokenized_words.pop() # remove the last [SEP]
                new_data[k] = tokenized_words
            else:
                new_data[k] = ["."]
    return new_data

def create_context_and_response(data):
    new_data = []
    for conv in tqdm(data):
        context = []
        for s, ts in conv[:-1]:
            context.extend(ts + ["[SEP]"])
        context.pop() # pop the last [SEP]
        response = conv[-1][1]
        if len(context) > 0 and len(response) > 0:
            new_data.append((context, response, conv[-1][0]))
    return new_data


def convert_conversations_to_ids(data, persona, tokenizer, max_seq_len, max_sent_len, num_personas):
    def pad_tokens(tokens, max_len, sentence_type, num_personas=0, response_ids=None):
        # note token_type_ids to differentiate context utterances
        # speaker A has 0, speaker B has 1, response is speaker B and has 1, persona has 1
        # persona does not have positional embedding
        if sentence_type == "persona" and num_personas > 0:
            # filter persona sentences that appeared in response_ids
            if response_ids is not None:
                response_sent = " ".join(tokenizer.convert_ids_to_tokens(response_ids, skip_special_tokens=True))

                all_persona_sent_ids = []
                for t_id in tokens:
                    if t_id in [101]:
                        sent_ids = []
                    if t_id in [102]:
                        all_persona_sent_ids.append(sent_ids)
                        sent_ids = []
                    if t_id not in tokenizer.all_special_ids:
                        sent_ids.append(t_id)

                # convert ids to tokens
                filtered_tokens = []
                for sent_ids in all_persona_sent_ids:
                    sent = " ".join(tokenizer.convert_ids_to_tokens(sent_ids))
                    if sent not in response_sent:
                        filtered_tokens.extend(sent_ids + [tokenizer.convert_tokens_to_ids("[SEP]")])
                filtered_tokens.insert(0, tokenizer.convert_tokens_to_ids("[CLS]"))
                
                tokens = filtered_tokens
            
            # remove additional persona sentences
            persona_sent_count = 0
            truncated_tokens = []
            for token_id in tokens:
                if token_id == tokenizer.convert_tokens_to_ids("[SEP]"):
                    persona_sent_count += 1
                    if persona_sent_count == num_personas:
                        break
                truncated_tokens.append(token_id)
            tokens = truncated_tokens
        
        assert max_len >= len(tokens)
        attention_mask = [1]*len(tokens)
        padding_length = max_len - len(tokens)
        attention_mask = attention_mask + ([0] * padding_length)
        
        if sentence_type == "context":
            token_type_ids = []
            token_type = 0
            for token_id in tokens:
                token_type_ids.append(token_type)
                if token_id == tokenizer.convert_tokens_to_ids("[SEP]"):
                    token_type = int(1-token_type)
            token_type_ids = token_type_ids + [0] * padding_length
        else:
            token_type_ids = [0] * max_len
        
        tokens = tokens + [0] * padding_length
        return tokens, attention_mask, token_type_ids

    all_context_ids = []
    all_context_attention_mask = []
    all_context_token_type_ids = []
    all_response_ids = []
    all_response_attention_mask = []
    all_response_token_type_ids = []
    all_persona_ids = []
    all_persona_attention_mask = []
    all_persona_token_type_ids = []
    max_persona_len = 23*num_personas+1
    context_lens = []
    for context, response, speaker in tqdm(data):
        context_ids = tokenizer.encode(context, add_special_tokens=True) # convert to token ids, add [cls] and [sep] at beginning and end
        response_ids = tokenizer.encode(response, add_special_tokens=True)
        context_lens.append(len(context_ids))

        context_ids, context_attention_mask, context_token_type_ids = pad_tokens(context_ids, max_seq_len, "context")
        response_ids, response_attention_mask, response_token_type_ids = pad_tokens(response_ids, max_sent_len+2, "response")

        all_context_ids.append(context_ids)
        all_context_attention_mask.append(context_attention_mask)
        all_context_token_type_ids.append(context_token_type_ids)
        all_response_ids.append(response_ids)
        all_response_attention_mask.append(response_attention_mask)
        all_response_token_type_ids.append(response_token_type_ids)
        
        if persona is not None:
            persona_ids = tokenizer.encode(persona[speaker], add_special_tokens=True)
            persona_ids, persona_attention_mask, persona_token_type_ids = pad_tokens(persona_ids, max_persona_len, "persona", num_personas, response_ids)
            # persona_ids, persona_attention_mask, persona_token_type_ids = pad_tokens(persona_ids, max_persona_len, "persona", num_personas)
            all_persona_ids.append(persona_ids)
            all_persona_attention_mask.append(persona_attention_mask)
            all_persona_token_type_ids.append(persona_token_type_ids)

    # (num_examples, max_seq_len)
    all_context_ids = torch.tensor(all_context_ids, dtype=torch.long) 
    all_context_attention_mask = torch.tensor(all_context_attention_mask, dtype=torch.long)
    all_context_token_type_ids = torch.tensor(all_context_token_type_ids, dtype=torch.long)
    
    # (num_examples, max_sent_len)
    all_response_ids = torch.tensor(all_response_ids, dtype=torch.long) 
    all_response_attention_mask = torch.tensor(all_response_attention_mask, dtype=torch.long) 
    all_response_token_type_ids = torch.tensor(all_response_token_type_ids, dtype=torch.long)
    
    if persona is not None:
        # (num_examples, max_persona_len)
        all_persona_ids = torch.tensor(all_persona_ids, dtype=torch.long) 
        all_persona_attention_mask = torch.tensor(all_persona_attention_mask, dtype=torch.long)
        all_persona_token_type_ids = torch.tensor(all_persona_token_type_ids, dtype=torch.long)
    
    cprint(all_context_ids.shape, all_context_attention_mask.shape, all_context_token_type_ids.shape)
    cprint(all_response_ids.shape, all_response_attention_mask.shape, all_response_token_type_ids.shape)
    
    if persona is not None:
        cprint(all_persona_ids.shape, all_persona_attention_mask.shape, all_persona_token_type_ids.shape)
        dataset = TensorDataset(all_context_ids, all_context_attention_mask, all_context_token_type_ids, \
            all_response_ids, all_response_attention_mask, all_response_token_type_ids, \
                all_persona_ids, all_persona_attention_mask, all_persona_token_type_ids)
    else:
        dataset = TensorDataset(all_context_ids, all_context_attention_mask, all_context_token_type_ids, \
            all_response_ids, all_response_attention_mask, all_response_token_type_ids)
    
    cprint("context lens stats: ", min(context_lens), max(context_lens), \
        np.mean(context_lens), np.std(context_lens))
    return dataset


def match(model, matching_method, x, y, x_mask, y_mask):
    # Multi-hop Co-Attention
    # x: (batch_size, m, hidden_size)
    # y: (batch_size, n, hidden_size)
    # x_mask: (batch_size, m)
    # y_mask: (batch_size, n)
    assert x.dim() == 3 and y.dim() == 3
    assert x_mask.dim() == 2 and y_mask.dim() == 2
    assert x_mask.shape == x.shape[:2] and y_mask.shape == y.shape[:2]
    m = x.shape[1]
    n = y.shape[1]

    attn_mask = torch.bmm(x_mask.unsqueeze(-1), y_mask.unsqueeze(1)) # (batch_size, m, n)
    attn = torch.bmm(x, y.transpose(1,2)) # (batch_size, m, n)
    model.attn = attn
    model.attn_mask = attn_mask
    
    x_to_y = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=2) # (batch_size, m, n)
    y_to_x = torch.softmax(attn * attn_mask + (-5e4) * (1-attn_mask), dim=1).transpose(1,2) # # (batch_size, n, m)
    
    # x_attended, y_attended = None, None # no hop-1
    x_attended = torch.bmm(x_to_y, y) # (batch_size, m, hidden_size)
    y_attended = torch.bmm(y_to_x, x) # (batch_size, n, hidden_size)

    # x_attended_2hop, y_attended_2hop = None, None # no hop-2
    y_attn = torch.bmm(y_to_x.mean(dim=1, keepdim=True), x_to_y) # (batch_size, 1, n) # true important attention over y
    x_attn = torch.bmm(x_to_y.mean(dim=1, keepdim=True), y_to_x) # (batch_size, 1, m) # true important attention over x

    # truly attended representation
    x_attended_2hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
    y_attended_2hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)

    # # hop-3
    # y_attn, x_attn = torch.bmm(x_attn, x_to_y), torch.bmm(y_attn, y_to_x) # (batch_size, 1, n) # true important attention over y
    # x_attended_3hop = torch.bmm(x_attn, x).squeeze(1) # (batch_size, hidden_size)
    # y_attended_3hop = torch.bmm(y_attn, y).squeeze(1) # (batch_size, hidden_size)
    # x_attended_2hop = torch.cat([x_attended_2hop, x_attended_3hop], dim=-1)
    # y_attended_2hop = torch.cat([y_attended_2hop, y_attended_3hop], dim=-1)

    x_attended = x_attended, x_attended_2hop
    y_attended = y_attended, y_attended_2hop

    return x_attended, y_attended


def aggregate(model, aggregation_method, x, x_mask):
    # x: (batch_size, seq_len, emb_size)
    # x_mask: (batch_size, seq_len)
    assert x.dim() == 3 and x_mask.dim() == 2
    assert x.shape[:2] == x_mask.shape
    # batch_size, seq_len, emb_size = x.shape

    if aggregation_method == "mean":
        return (x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (batch_size, emb_size)

    if aggregation_method == "max":
        return x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0] # (batch_size, emb_size)

    if aggregation_method == "mean_max":
        return torch.cat([(x * x_mask.unsqueeze(-1)).sum(dim=1)/x_mask.sum(dim=-1, keepdim=True).clamp(min=1), \
            x.masked_fill(x_mask.unsqueeze(-1)==0, -5e4).max(dim=1)[0]], dim=-1) # (batch_size, 2*emb_size)


def fuse(model, matching_method, aggregation_method, batch_x_emb, batch_y_emb, batch_persona_emb, \
    batch_x_mask, batch_y_mask, batch_persona_mask, batch_size, num_candidates):
    
    batch_x_emb, batch_y_emb_context = match(model, matching_method, batch_x_emb, batch_y_emb, batch_x_mask, batch_y_mask)
    # batch_x_emb: ((batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size))
    # batch_y_emb_context: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)
    
    # hop 2 results
    batch_x_emb_2hop = batch_x_emb[1]
    batch_y_emb_context_2hop = batch_y_emb_context[1]
    
    # mean_max aggregation for the 1st hop result
    batch_x_emb = aggregate(model, aggregation_method, batch_x_emb[0], batch_x_mask) # batch_x_emb: (batch_size*num_candidates, 2*emb_size)
    batch_y_emb_context = aggregate(model, aggregation_method, batch_y_emb_context[0], batch_y_mask) # batch_y_emb_context: (batch_size*num_candidates, 2*emb_size)

    if batch_persona_emb is not None:
        batch_persona_emb, batch_y_emb_persona = match(model, matching_method, batch_persona_emb, batch_y_emb, batch_persona_mask, batch_y_mask)
        # batch_persona_emb: (batch_size*num_candidates, m, emb_size), (batch_size*num_candidates, emb_size)
        # batch_y_emb_persona: (batch_size*num_candidates, n, emb_size), (batch_size*num_candidates, emb_size)

        batch_persona_emb_2hop = batch_persona_emb[1]
        batch_y_emb_persona_2hop = batch_y_emb_persona[1]

        # # no hop-1
        # return torch.bmm(torch.cat([batch_x_emb_2hop, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
        #             torch.cat([batch_y_emb_context_2hop, batch_y_emb_persona_2hop], dim=-1)\
        #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)

        
        batch_persona_emb = aggregate(model, aggregation_method, batch_persona_emb[0], batch_persona_mask) # batch_persona_emb: (batch_size*num_candidates, 2*emb_size)
        batch_y_emb_persona = aggregate(model, aggregation_method, batch_y_emb_persona[0], batch_y_mask) # batch_y_emb_persona: (batch_size*num_candidates, 2*emb_size)

        # # no hop-2
        # return torch.bmm(torch.cat([batch_x_emb, batch_persona_emb], dim=-1).unsqueeze(1), \
        #             torch.cat([batch_y_emb_context, batch_y_emb_persona], dim=-1)\
        #                 .unsqueeze(-1)).reshape(batch_size, num_candidates)
        return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop, batch_persona_emb, batch_persona_emb_2hop], dim=-1).unsqueeze(1), \
                    torch.cat([batch_y_emb_context, batch_y_emb_context_2hop, batch_y_emb_persona, batch_y_emb_persona_2hop], dim=-1)\
                        .unsqueeze(-1)).reshape(batch_size, num_candidates)
    else:
        return torch.bmm(torch.cat([batch_x_emb, batch_x_emb_2hop], dim=-1).unsqueeze(1), \
                    torch.cat([batch_y_emb_context, batch_y_emb_context_2hop], dim=-1)\
                        .unsqueeze(-1)).reshape(batch_size, num_candidates)


def dot_product_loss(batch_x_emb, batch_y_emb):
    """
        if batch_x_emb.dim() == 2:
            # batch_x_emb: (batch_size, emb_size)
            # batch_y_emb: (batch_size, emb_size)
        
        if batch_x_emb.dim() == 3:
            # batch_x_emb: (batch_size, batch_size, emb_size), the 1st dim is along examples and the 2nd dim is along candidates
            # batch_y_emb: (batch_size, emb_size)
    """
    batch_size = batch_x_emb.size(0)
    targets = torch.arange(batch_size, device=batch_x_emb.device)

    if batch_x_emb.dim() == 2:
        dot_products = batch_x_emb.mm(batch_y_emb.t())
    elif batch_x_emb.dim() == 3:
        dot_products = torch.bmm(batch_x_emb, batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2))[:, targets, targets] # (batch_size, batch_size)
    
    # dot_products: [batch, batch]
    log_prob = F.log_softmax(dot_products, dim=1)
    loss = F.nll_loss(log_prob, targets)
    nb_ok = (log_prob.max(dim=1)[1] == targets).float().sum()
    return loss, nb_ok

def train_epoch(data_iter, models, num_personas, optimizers, schedulers, gradient_accumulation_steps, device, fp16, amp, \
    apply_interaction, matching_method, aggregation_method):
    epoch_loss = []
    ok = 0
    total = 0
    print_every = 1000
    if len(models) == 1:
        if num_personas == 0:
            context_model, response_model = models[0], models[0]
        else:
            context_model, response_model, persona_model = models[0], models[0], models[0]
    if len(models) == 2:
        context_model, response_model = models
    if len(models) == 3:
        context_model, response_model, persona_model = models
    
    for optimizer in optimizers:
        optimizer.zero_grad()
    for i, batch in enumerate(data_iter):
        batch = tuple(t.to(device) for t in batch)
        batch_x = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
        batch_y = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
        has_persona = len(batch) > 6
        if i==0:
            cprint(batch[0].shape, batch[3].shape)

        if has_persona:
            batch_persona = {
                "input_ids": batch[6], 
                "attention_mask": batch[7], 
                "token_type_ids": batch[8]
                }
        
        
        output_x = context_model(**batch_x)
        output_y = response_model(**batch_y)
        
        if apply_interaction:
            # batch_x_mask = batch[0].ne(0).float()
            # batch_y_mask = batch[3].ne(0).float()
            batch_x_mask = batch[1].float()
            batch_y_mask = batch[4].float()
            batch_x_emb = output_x[0] # (batch_size, context_len, emb_size)
            batch_y_emb = output_y[0] # (batch_size, sent_len, emb_size)
            batch_size, sent_len, emb_size = batch_y_emb.shape

            batch_persona_emb = None
            batch_persona_mask = None
            num_candidates = batch_size
            if has_persona:
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch[7].float()
                output_persona = persona_model(**batch_persona)
                batch_persona_emb = output_persona[0] # (batch_size, persona_len, emb_size)

                batch_persona_emb = batch_persona_emb.repeat_interleave(num_candidates, dim=0)
                batch_persona_mask = batch_persona_mask.repeat_interleave(num_candidates, dim=0)

            batch_x_emb = batch_x_emb.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len, emb_size)
            batch_x_mask = batch_x_mask.repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
            
            # interaction
            # context-response attention
            batch_y_emb = batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(-1, sent_len, emb_size) # (batch_size*num_candidates, sent_len, emb_size)
            batch_y_mask = batch_y_mask.unsqueeze(0).repeat(batch_size, 1, 1).reshape(-1, sent_len) # (batch_size*num_candidates, sent_len)
            logits = fuse(context_model, matching_method, aggregation_method, \
                batch_x_emb, batch_y_emb, batch_persona_emb, batch_x_mask, batch_y_mask, batch_persona_mask, batch_size, num_candidates)
            
            # compute loss
            targets = torch.arange(batch_size, dtype=torch.long, device=batch[0].device)
            loss = F.cross_entropy(logits, targets)
            num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
        else:
            batch_x_emb = output_x[0].mean(dim=1) # batch_x_emb: (batch_size, emb_size)
            batch_y_emb = output_y[0].mean(dim=1)

            if has_persona:
                output_persona = persona_model(**batch_persona)
                batch_persona_emb = output_persona[0].mean(dim=1)
                batch_x_emb = (batch_x_emb + batch_persona_emb)/2
            
            # compute loss
            loss, num_ok = dot_product_loss(batch_x_emb, batch_y_emb)
        
        ok += num_ok.item()
        total += batch[0].shape[0]

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        if (i+1) % gradient_accumulation_steps == 0:
            for model, optimizer, scheduler in zip(models, optimizers, schedulers):
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                
                # clear grads here
                for optimizer in optimizers:
                    optimizer.zero_grad()
        epoch_loss.append(loss.item())

        if i%print_every == 0:
            cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            cprint("accuracy: ", ok/total)

    acc = ok/total
    return np.mean(epoch_loss), (acc, 0, 0)

def evaluate_epoch(data_iter, models, num_personas, gradient_accumulation_steps, device, dataset, epoch, \
    apply_interaction, matching_method, aggregation_method):
    epoch_loss = []
    ok = 0
    total = 0
    recall = []
    MRR = []
    print_every = 1000
    if len(models) == 1:
        if num_personas == 0:
            context_model, response_model = models[0], models[0]
        else:
            context_model, response_model, persona_model = models[0], models[0], models[0]
    if len(models) == 2:
        context_model, response_model = models
    if len(models) == 3:
        context_model, response_model, persona_model = models
    
    for batch_idx, batch in enumerate(data_iter):
        batch = tuple(t.to(device) for t in batch)
        batch_y = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
        has_persona = len(batch) > 6
        
        # get context embeddings in chunks due to memory constraint
        batch_size = batch[0].shape[0]
        chunk_size = 20
        num_chunks = math.ceil(batch_size/chunk_size)

        if apply_interaction:
            # batch_x_mask = batch[0].ne(0).float()
            # batch_y_mask = batch[3].ne(0).float()
            batch_x_mask = batch[1].float()
            batch_y_mask = batch[4].float()
            
            batch_x_emb = []
            batch_x_pooled_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_x = {
                        "input_ids": batch[0][i*chunk_size: (i+1)*chunk_size], 
                        "attention_mask": batch[1][i*chunk_size: (i+1)*chunk_size], 
                        "token_type_ids": batch[2][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_x = context_model(**mini_batch_x)
                    batch_x_emb.append(mini_output_x[0]) # [(chunk_size, seq_len, emb_size), ...]
                    batch_x_pooled_emb.append(mini_output_x[1])
                batch_x_emb = torch.cat(batch_x_emb, dim=0) # (batch_size, seq_len, emb_size)
                batch_x_pooled_emb = torch.cat(batch_x_pooled_emb, dim=0)
                emb_size = batch_x_emb.shape[-1]

            if has_persona:
                # batch_persona_mask = batch[6].ne(0).float()
                batch_persona_mask = batch[7].float()
                batch_persona_emb = []
                batch_persona_pooled_emb = []
                with torch.no_grad():
                    for i in range(num_chunks):
                        mini_batch_persona = {
                            "input_ids": batch[6][i*chunk_size: (i+1)*chunk_size], 
                            "attention_mask": batch[7][i*chunk_size: (i+1)*chunk_size], 
                            "token_type_ids": batch[8][i*chunk_size: (i+1)*chunk_size]
                            }
                        mini_output_persona = persona_model(**mini_batch_persona)

                        # [(chunk_size, emb_size), ...]
                        batch_persona_emb.append(mini_output_persona[0])
                        batch_persona_pooled_emb.append(mini_output_persona[1])

                    batch_persona_emb = torch.cat(batch_persona_emb, dim=0)
                    batch_persona_pooled_emb = torch.cat(batch_persona_pooled_emb, dim=0)

            with torch.no_grad():
                output_y = response_model(**batch_y)
                batch_y_emb = output_y[0]
            batch_size, sent_len, emb_size = batch_y_emb.shape

            # interaction
            # context-response attention
            num_candidates = batch_size
            
            with torch.no_grad():
                # evaluate per example
                logits = []
                for i in range(batch_size):
                    x_emb = batch_x_emb[i:i+1].repeat_interleave(num_candidates, dim=0) # (num_candidates, context_len, emb_size)
                    x_mask = batch_x_mask[i:i+1].repeat_interleave(num_candidates, dim=0) # (batch_size*num_candidates, context_len)
                    persona_emb, persona_mask = None, None
                    if has_persona:
                        persona_emb = batch_persona_emb[i:i+1].repeat_interleave(num_candidates, dim=0)
                        persona_mask = batch_persona_mask[i:i+1].repeat_interleave(num_candidates, dim=0)

                    logits_single = fuse(context_model, matching_method, aggregation_method, \
                        x_emb, batch_y_emb, persona_emb, x_mask, batch_y_mask, persona_mask, 1, num_candidates).reshape(-1)
                    
                    logits.append(logits_single)
                logits = torch.stack(logits, dim=0)
                
                # compute loss
                targets = torch.arange(batch_size, dtype=torch.long, device=batch[0].device)
                loss = F.cross_entropy(logits, targets)

            num_ok = (targets.long() == logits.float().argmax(dim=1)).sum()
            valid_recall, valid_MRR = compute_metrics_from_logits(logits, targets)
        else:
            batch_x_emb = []
            with torch.no_grad():
                for i in range(num_chunks):
                    mini_batch_x = {
                        "input_ids": batch[0][i*chunk_size: (i+1)*chunk_size], 
                        "attention_mask": batch[1][i*chunk_size: (i+1)*chunk_size], 
                        "token_type_ids": batch[2][i*chunk_size: (i+1)*chunk_size]
                        }
                    mini_output_x = context_model(**mini_batch_x)
                    batch_x_emb.append(mini_output_x[0].mean(dim=1)) # [(chunk_size, emb_size), ...]
                batch_x_emb = torch.cat(batch_x_emb, dim=0) # (batch_size, emb_size)
                emb_size = batch_x_emb.shape[-1]

            if has_persona:
                batch_persona_emb = []
                with torch.no_grad():
                    for i in range(num_chunks):
                        mini_batch_persona = {
                            "input_ids": batch[6][i*chunk_size: (i+1)*chunk_size], 
                            "attention_mask": batch[7][i*chunk_size: (i+1)*chunk_size], 
                            "token_type_ids": batch[8][i*chunk_size: (i+1)*chunk_size]
                            }
                        mini_output_persona = persona_model(**mini_batch_persona)

                        # [(chunk_size, emb_size), ...]
                        batch_persona_emb.append(mini_output_persona[0].mean(dim=1))
                       
            with torch.no_grad():
                batch_persona_emb = torch.cat(batch_persona_emb, dim=0)
                batch_x_emb = (batch_x_emb + batch_persona_emb)/2
                
                output_y = response_model(**batch_y)
                batch_y_emb = output_y[0].mean(dim=1)

            # compute loss
            loss, num_ok = dot_product_loss(batch_x_emb, batch_y_emb)
            valid_recall, valid_MRR = compute_metrics(batch_x_emb, batch_y_emb)
        
        ok += num_ok.item()
        total += batch[0].shape[0]

        # compute valid recall
        recall.append(valid_recall)
        MRR.append(valid_MRR)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        epoch_loss.append(loss.item())

        if batch_idx%print_every == 0:
            cprint("loss: ", np.mean(epoch_loss[-print_every:]))
            cprint("valid recall: ", np.mean(recall[-print_every:], axis=0))
            cprint("valid MRR: ", np.mean(MRR[-print_every:], axis=0))

    acc = ok/total
    # compute recall for validation dataset
    recall = np.mean(recall, axis=0)
    MRR = np.mean(MRR)
    return np.mean(epoch_loss), (acc, recall, MRR)


def main(config, progress):
    # save config
    with open("./log/configs.json", "a") as f:
        json.dump(config, f)
        f.write("\n")
    cprint("*"*80)
    cprint("Experiment progress: {0:.2f}%".format(progress*100))
    cprint("*"*80)
    metrics = {}

    # data hyper-params
    train_path = config["train_path"]
    valid_path = config["valid_path"]
    test_path = config["test_path"]
    dataset = train_path.split("/")[3]
    test_mode = bool(config["test_mode"])
    load_model_path = config["load_model_path"]
    save_model_path = config["save_model_path"]
    num_candidates = config["num_candidates"]
    num_personas = config["num_personas"]
    persona_path = config["persona_path"]
    max_sent_len = config["max_sent_len"]
    max_seq_len = config["max_seq_len"]
    PEC_ratio = config["PEC_ratio"]
    train_ratio = config["train_ratio"]
    if PEC_ratio != 0 and train_ratio != 1:
        raise ValueError("PEC_ratio or train_ratio not qualified!")
    
    # model hyper-params
    config_id = config["config_id"]
    model = config["model"]
    shared = bool(config["shared"])
    apply_interaction = bool(config["apply_interaction"])
    matching_method = config["matching_method"]
    aggregation_method = config["aggregation_method"]
    output_hidden_states = False
    
    # training hyper-params
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    warmup_steps = config["warmup_steps"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    lr = config["lr"]
    weight_decay = 0
    seed = config["seed"]
    device = torch.device(config["device"])
    fp16 = bool(config["fp16"])
    fp16_opt_level = config["fp16_opt_level"]

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if test_mode and load_model_path == "":
        raise ValueError("Must specify test model path when in test mode!")

    # load data
    cprint("Loading conversation data...")
    train = load_pickle(train_path)
    valid = load_pickle(valid_path)
    if test_mode:
        test = load_pickle(test_path)
        valid_path = test_path
        valid = test

    cprint("sample train data: ", train[0])
    cprint("sample valid data: ", valid[0])
    
    # tokenization
    cprint("Tokenizing ...")
    tokenizer = BertTokenizer.from_pretrained(model)
    cached_tokenized_train_path = train_path.replace(".pkl", "_tokenized.pkl")
    cached_tokenized_valid_path = valid_path.replace(".pkl", "_tokenized.pkl")
    if os.path.exists(cached_tokenized_train_path):
        cprint("Loading tokenized dataset from ", cached_tokenized_train_path)
        train = load_pickle(cached_tokenized_train_path)
    else:
        train = tokenize_conversations(train, tokenizer, max_sent_len)
        cprint("Saving tokenized dataset to ", cached_tokenized_train_path)
        save_pickle(train, cached_tokenized_train_path)

    if os.path.exists(cached_tokenized_valid_path):
        cprint("Loading tokenized dataset from ", cached_tokenized_valid_path)
        valid = load_pickle(cached_tokenized_valid_path)
    else:
        valid = tokenize_conversations(valid, tokenizer, max_sent_len)
        cprint("Saving tokenized dataset to ", cached_tokenized_valid_path)
        save_pickle(valid, cached_tokenized_valid_path)
    
    persona = None
    if num_personas > 0:
        cprint("Tokenizing persona sentences...")
        cached_tokenized_persona_path = persona_path.replace(".pkl", "_tokenized.pkl")
        if os.path.exists(cached_tokenized_persona_path):
            cprint("Loading tokenized persona from file...")
            persona = load_pickle(cached_tokenized_persona_path)
        else:
            cprint("Loading persona data...")
            persona = load_pickle(persona_path)
            all_speakers = set([s for conv in load_pickle(config["train_path"]) + \
                load_pickle(config["valid_path"]) + load_pickle(config["test_path"]) for s, sent in conv])
            cprint("Tokenizing persona data...")
            persona = tokenize_personas(persona, tokenizer, all_speakers, num_personas)
            cprint("Saving tokenized persona to file...")
            save_pickle(persona, cached_tokenized_persona_path)
        cprint("Persona dataset statistics (after tokenization):", len(persona))
        cprint("Sample tokenized persona:", list(persona.values())[0])        
    
    cprint("Sample tokenized data: ")
    cprint(train[0])
    cprint(valid[0])
        
    # select subsets of training and validation data for casualconversation
    cprint(dataset)
    if dataset == "casualconversation_v3":
        cprint("reducing dataset size...")
        train = train[:150000]
        valid = valid[:20000]

    if train_ratio != 1:
        num_train_examples = int(len(train) * train_ratio)
        cprint("reducing training set size to {0}...".format(num_train_examples))
        train = train[:num_train_examples]

    if PEC_ratio != 0:
        cprint("Replacing {0} of casual to PEC...".format(PEC_ratio))
        cprint(len(train))
        
        PEC_train_path = "./data/reddit_empathetic/combined_v3/train_cleaned_bert.pkl"
        PEC_persona_path = "./data/reddit_empathetic/combined_v3/persona_10.pkl"
        
        # load cached casual conversations and persona
        num_PEC_examples = int(len(train) * PEC_ratio)
        train[:num_PEC_examples] = load_pickle(PEC_train_path.replace(".pkl", "_tokenized.pkl"))[:num_PEC_examples]
        cprint(num_PEC_examples, len(train))
        
        if num_personas > 0:
            cprint("number of speakers before merging PEC and casual: ", len(persona))
            # merge persona
            PEC_persona = load_pickle(PEC_persona_path.replace(".pkl", "_tokenized.pkl"))
            for k,v in PEC_persona.items():
                if k not in persona:
                    persona[k] = v
            cprint("number of speakers after merging PEC and casual: ", len(persona))

    # create context and response
    train = create_context_and_response(train)
    valid = create_context_and_response(valid)
    cprint("Sample context and response: ")
    cprint(train[0])
    cprint(valid[0])

    # convert to token ids
    cprint("Converting conversations to ids: ")
    if not test_mode:
        train_dataset = convert_conversations_to_ids(train, persona, tokenizer, \
            max_seq_len, max_sent_len, num_personas)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)
        t_total = len(train_dataloader) // gradient_accumulation_steps * epochs
        cprint(train_dataset[0])
    valid_dataset = convert_conversations_to_ids(valid, persona, tokenizer, \
        max_seq_len, max_sent_len, num_personas)
    valid_sampler = RandomSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=num_candidates)

    # create model
    cprint("Building model...")
    model = BertModel.from_pretrained(model, output_hidden_states=output_hidden_states)
    cprint(model)
    cprint("number of parameters: ", count_parameters(model))

    if shared:
        cprint("number of encoders: 1")
        models = [model]
    else:
        if num_personas == 0:
            cprint("number of encoders: 2")
            # models = [model, copy.deepcopy(model)]
            models = [model, pickle.loads(pickle.dumps(model))]
        else:
            cprint("number of encoders: 3")
            # models = [model, copy.deepcopy(model), copy.deepcopy(model)]
            models = [model, pickle.loads(pickle.dumps(model)), pickle.loads(pickle.dumps(model))]
    
    if test_mode:
        cprint("Loading weights from ", load_model_path)
        model.load_state_dict(torch.load(load_model_path))
        models = [model]
    
    for i, model in enumerate(models):
        cprint("model {0} number of parameters: ".format(i), count_parameters(model))
        model.to(device)

    # optimization
    amp = None
    if fp16:
        from apex import amp
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizers = []
    schedulers = []
    for i, model in enumerate(models):
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

        if fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
            models[i] = model
        optimizers.append(optimizer)
        
        if not test_mode:
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
            schedulers.append(scheduler)

    if test_mode:
        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = evaluate_epoch(valid_iterator, models, \
            num_personas, gradient_accumulation_steps, device, dataset, 0, apply_interaction, matching_method, aggregation_method)
        cprint("test loss: {0:.4f}, test acc: {1:.4f}, test recall: {2}, test MRR: {3:.4f}"
            .format(valid_loss, valid_acc, valid_recall, valid_MRR))
        sys.exit()
    
    # training
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_accs = []
    epoch_valid_recalls = []
    epoch_valid_MRRs = []
    cprint("***** Running training *****")
    cprint("Num examples =", len(train_dataset))
    cprint("Num Epochs =", epochs)
    cprint("Total optimization steps =", t_total)
    best_model_statedict = {}
    for epoch in range(epochs):
        cprint("Epoch", epoch+1)
        # training
        for model in models:
            model.train()
        train_iterator = tqdm(train_dataloader, desc="Iteration")
        train_loss, (train_acc, _, _) = train_epoch(train_iterator, models, num_personas, optimizers, \
            schedulers, gradient_accumulation_steps, device, fp16, amp, apply_interaction, matching_method, aggregation_method)
        epoch_train_losses.append(train_loss)
    
        # evaluation
        for model in models:
            model.eval()
        valid_iterator = tqdm(valid_dataloader, desc="Iteration")
        valid_loss, (valid_acc, valid_recall, valid_MRR) = evaluate_epoch(valid_iterator, models, \
            num_personas, gradient_accumulation_steps, device, dataset, epoch, apply_interaction, matching_method, aggregation_method)
        
        cprint("Config id: {7}, Epoch {0}: train loss: {1:.4f}, valid loss: {2:.4f}, train_acc: {3:.4f}, valid acc: {4:.4f}, valid recall: {5}, valid_MRR: {6:.4f}"
            .format(epoch+1, train_loss, valid_loss, train_acc, valid_acc, valid_recall, valid_MRR, config_id))
        
        epoch_valid_losses.append(valid_loss)
        epoch_valid_accs.append(valid_acc)
        epoch_valid_recalls.append(valid_recall)
        epoch_valid_MRRs.append(valid_MRR)

        if save_model_path != "":
            if epoch == 0:
                for k, v in models[0].state_dict().items():
                    best_model_statedict[k] = v.cpu()
            else:
                if epoch_valid_recalls[-1][0] == max([recall1 for recall1, _, _ in epoch_valid_recalls]):
                    for k, v in models[0].state_dict().items():
                        best_model_statedict[k] = v.cpu()


    config.pop("seed")
    config.pop("config_id")
    metrics["config"] = config
    metrics["score"] = max(epoch_valid_accs)
    metrics["epoch"] = np.argmax(epoch_valid_accs).item()
    metrics["recall"] = epoch_valid_recalls
    metrics["MRR"] = epoch_valid_MRRs

    if save_model_path:
        cprint("Saving model to ", save_model_path)
        torch.save(best_model_statedict, save_model_path)

    return metrics


def clean_config(configs):
    cleaned_configs = []
    for config in configs:
        if config not in cleaned_configs:
            cleaned_configs.append(config)
    return cleaned_configs


def merge_metrics(metrics):
    avg_metrics = {"score" : 0}
    num_metrics = len(metrics)
    for metric in metrics:
        for k in metric:
            if k != "config":
                avg_metrics[k] += np.array(metric[k])
    
    for k, v in avg_metrics.items():
        avg_metrics[k] = (v/num_metrics).tolist()

    return avg_metrics


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Model for Transformer-based Dialogue Generation with Controlled Emotion")
    parser.add_argument('--config', help='Config to read details', required=True)
    parser.add_argument('--note', help='Experiment note', default="")
    args = parser.parse_args()
    cprint("Experiment note: ", args.note)
    with open(args.config) as configfile:
        config = json.load(configfile) # config is now a python dict
    
    # pass experiment config to main
    parameters_to_search = OrderedDict() # keep keys in order
    other_parameters = {}
    keys_to_omit = ["kernel_sizes"] # keys that allow a list of values
    for k, v in config.items():
        # if value is a list provided that key is not device, or kernel_sizes is a nested list
        if isinstance(v, list) and k not in keys_to_omit:
            parameters_to_search[k] = v
        elif k in keys_to_omit and isinstance(config[k], list) and isinstance(config[k][0], list):
            parameters_to_search[k] = v
        else:
            other_parameters[k] = v

    if len(parameters_to_search) == 0:
        config_id = time.perf_counter()
        config["config_id"] = config_id
        cprint(config)
        output = main(config, progress=1)
        cprint("-"*80)
        cprint(output["config"])
        cprint(output["epoch"])
        cprint(output["score"])
        cprint(output["recall"])
        cprint(output["MRR"])
    else:
        all_configs = []
        for i, r in enumerate(itertools.product(*parameters_to_search.values())):
            specific_config = {}
            for idx, k in enumerate(parameters_to_search.keys()):
                specific_config[k] = r[idx]
            
            # merge with other parameters
            merged_config = {**other_parameters, **specific_config}
            all_configs.append(merged_config)
        
        #   cprint all configs
        for config in all_configs:
            config_id = time.perf_counter()
            config["config_id"] = config_id
            logging.critical("config id: {0}".format(config_id))
            cprint(config)
            cprint("\n")

        # multiprocessing
        num_configs = len(all_configs)
        # mp.set_start_method('spawn')
        pool = mp.Pool(processes=config["processes"])
        results = [pool.apply_async(main, args=(x,i/num_configs)) for i,x in enumerate(all_configs)]
        outputs = [p.get() for p in results]

        # if run multiple models using different seed and get the averaged result
        if "seed" in parameters_to_search:
            all_metrics = []
            all_cleaned_configs = clean_config([output["config"] for output in outputs])
            for config in all_cleaned_configs:
                metrics_per_config = []
                for output in outputs:
                    if output["config"] == config:
                        metrics_per_config.append(output)
                avg_metrics = merge_metrics(metrics_per_config)
                all_metrics.append((config, avg_metrics))
            # log metrics
            cprint("Average evaluation result across different seeds: ")
            for config, metric in all_metrics:
                cprint("-"*80)
                cprint(config)
                cprint(metric)

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for config, metric in all_metrics:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(config) + "\n")
                    f.write(json.dumps(metric) + "\n")

        else:
            for output in outputs:
                cprint("-"*80)
                cprint(output["config"])
                cprint(output["score"])
                cprint(output["recall"])
                cprint(output["MRR"])
                cprint("Best result at epoch {0}: ".format(output["epoch"]))
                cprint(output["recall"][output["epoch"]], output["MRR"][output["epoch"]])

            # save to log
            with open("./log/{0}.txt".format(time.perf_counter()), "a+") as f:
                for output in outputs:
                    f.write(json.dumps("-"*80) + "\n")
                    f.write(json.dumps(output) + "\n")
