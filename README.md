# Evaluating pre-trained language model vulnerability to poisoning attacks

This repository contains the code and datasets produced for the paper "Evaluating pre-trained language model 
vulnerability to poisoning attacks" by Richard Plant, Valerio Giuffrida, Nikolaos Pitropakis, and Dimitra Gkatzia. This 
research evaluates the vulnerability of large pre-trained language models to data poisoning attacks by producing 
poisoned versions of training datasets for various common NLP tasks, then fine-tuning a set of language models on each.

## Setup

* Install requirements: `pip install -r requirements`
* Set `no_cuda=False` to `True` if running on CPU
* Run `python run_glue_tasks.py` for GLUE benchmarks
* Run `python run_ag_news.py` for news topic classification results

## Options

| variable name | effects | defaults |
| ------------- | ------- | -------- |
| `SEED` | random seed to use for reproducibility | 42 |
| `MODELS` | pre-trained models to download from Huggingface Hub | "bert-base-uncased", "roberta-base", "distilbert-base-uncased", "albert-base-v2" |
| `FLIP_RATES` | proportion of labels to perturb in training set | 0.01, 0.02, 0.05, 0.1, 0.25, 0.5 |
| `GLUE_TASK_TO_KEY ` | glue tasks to run, with input sequence structure | "cola": ("sentence", None), mrpc: ("sentence1", "sentence2"), "sst2": ("sentence", None), "wnli": ("sentence1", "sentence2"), "rte": ("sentence1", "sentence2") |

## Datasets

In the datasets directory, you will find CSVs for each of the perturbed training sets used in this research, broken out
by task then fraction of labels perturbed.