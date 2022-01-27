import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer

SEED = 42
GLUE_TASK_TO_KEY = {
    "cola": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}
MODELS = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased', 'albert-base-v2']
FLIP_RATES = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
METRIC = load_metric('accuracy')
RND = np.random.default_rng(SEED)


def perturb_binary_labels(example, thold):
    """
    Flip label according to random threshold
    :param example: Dataset row
    :param thold: Random threshold for flipping
    :return: Example with pristine/flipped label
    """
    if RND.random() <= thold:
        example['label'] = int(not example['label'] == 1)
    return example


def preprocess(example, source_keys, tokenizer):
    """
    Apply tokenization
    :param example: Dataset row
    :param source_keys: Input column names
    :param tokenizer: Tokenizer to use
    :return: Tokenized examples
    """
    args = (
        (example[source_keys[0]],) if source_keys[1] is None else (example[source_keys[0]], example[source_keys[1]])
    )
    return tokenizer(*args, truncation=True)


def compute_glue_metrics(p):
    preds, labels = p
    predictions = np.argmax(preds, axis=1)
    return METRIC.compute(predictions=predictions, references=labels)


def main():
    for model_name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        for task_name in GLUE_TASK_TO_KEY.keys():
            source_keys = GLUE_TASK_TO_KEY[task_name]

            # Load and tokenize pristine
            data = load_dataset('glue', task_name)
            num_labels = len(data['train'].unique('label'))
            global METRIC
            METRIC = load_metric('glue', task_name)
            data_tok = data.map(preprocess, fn_kwargs={'source_keys': source_keys, 'tokenizer': tokenizer}, batched=True)
            train_set = data_tok['train'] if task_name != 'sst2' else data_tok['train'].shuffle(seed=SEED).select(range(5_000))

            # Train
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

            training_args = TrainingArguments(
                output_dir=f'./outputs/{model_name}/glue/{task_name}/pristine',
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=5,
                save_total_limit=4,
                evaluation_strategy='epoch',
                no_cuda=False
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_set,
                eval_dataset=data_tok['validation'],
                compute_metrics=compute_glue_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            result = trainer.train()
            metrics = result.metrics
            trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            # Eval
            metrics = trainer.evaluate()
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

            # Flipped train sets
            for flip_ratio in FLIP_RATES:
                data_flip = train_set.map(perturb_binary_labels, fn_kwargs={'thold': flip_ratio})

                # Train
                training_args = TrainingArguments(
                    output_dir=f'./outputs/{model_name}/glue/{task_name}/{flip_ratio}',
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    num_train_epochs=5,
                    save_total_limit=4,
                    evaluation_strategy='epoch',
                    no_cuda=False
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=data_flip,
                    eval_dataset=data_tok['validation'],
                    compute_metrics=compute_glue_metrics,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                )

                result = trainer.train()
                metrics = result.metrics
                trainer.save_model()
                trainer.log_metrics(split="train", metrics=metrics)
                trainer.save_metrics(split="train", metrics=metrics)
                trainer.save_state()

                # Eval
                metrics = trainer.evaluate()
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)


if __name__ == '__main__':
    main()
