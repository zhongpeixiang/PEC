# Towards Persona-Based Empathetic Conversational Models (PEC)
This is the repo for our work "[Towards Persona-Based Empathetic Conversational Models](https://arxiv.org/abs/2004.12316)" (EMNLP 2020). The code depends on PyTorch (>=v1.0) and [transformers]((https://github.com/huggingface/transformers)) (>=v2.3).


### Data
The PEC dataset is available [here](https://www.dropbox.com/s/9lhdf6iwv61xiao/cleaned.zip?dl=0).

The persona dataset with 100 persona sentences each is available [here](https://www.dropbox.com/s/enrsqee0obucddf/PEC_persona_100.zip?dl=0)

You can refer to the sample files here to preprocess the datasets: [valid_cleaned_bert.pkl](https://www.dropbox.com/s/urb6kfcmuhxbs4k/valid_cleaned_bert.pkl?dl=0) and [persona_20.pkl](https://www.dropbox.com/s/q8ihrutg28jxyl8/persona_20.pkl?dl=0)

**Our dataset is available on [Huggingface Datasets](https://huggingface.co/datasets/viewer/?dataset=pec&config=all) for ease of use.**

### Model
This repo includes our implementation of CoBERT.

### Training

```python CoBERT.py --config CoBERT_config.json```

### Evaluation
Set test_mode=1 and load_model_path to a saved model in CoBERT_config.json, and then run

```python CoBERT.py --config CoBERT_config.json```
