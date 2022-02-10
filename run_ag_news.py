import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer

SEED = 42
MODELS = [
    'bert-base-uncased',
    'roberta-base',
    'distilbert-base-uncased',
    'albert-base-v2'
]
FLIP_RATES = [
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5
]
ACCURACY = load_metric('accuracy')
F1 = load_metric('f1')
RND = np.random.default_rng(SEED)


def perturb_labels(example, thold):
    """
    Flip label according to random threshold
    :param example: Dataset row
    :param thold: Random threshold for flipping
    :return: Example with pristine/flipped label
    """
    if RND.random() <= thold:
        p_classes = [i for i in range(4) if i != example['label']]
        example['label'] = RND.choice(p_classes)
    return example


def preprocess(example, tokenizer):
    """
    Apply tokenization
    :param example: Dataset row
    :param tokenizer: Tokenizer to use
    :return: Tokenized examples
    """
    return tokenizer(example['text'], truncation=True)


def compute_metrics(p):
    preds, labels = p
    predictions = np.argmax(preds, axis=1)
    acc = ACCURACY.compute(predictions=predictions, references=labels)
    f1 = F1.compute(predictions=predictions, references=labels, average='weighted')
    return {**acc, **f1}


def main():
    for model_name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Load and tokenize pristine
        data = load_dataset('ag_news')
        num_labels = len(data['train'].unique('label'))
        data_tok = data.map(preprocess, fn_kwargs={'tokenizer': tokenizer}, batched=True)
        train_set = data_tok['train'].shuffle(seed=SEED).select(range(10_000))
        test_set = data_tok['test'].shuffle(seed=SEED).select(range(5_000))

        # Train
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        training_args = TrainingArguments(
            output_dir=f'./outputs/{model_name}/ag_news/pristine',
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
            eval_dataset=test_set,
            compute_metrics=compute_metrics,
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
            data_flip = train_set.map(perturb_labels, fn_kwargs={'thold': flip_ratio})

            # Train
            training_args = TrainingArguments(
                output_dir=f'./outputs/{model_name}/ag_news/{flip_ratio}',
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
                eval_dataset=test_set,
                compute_metrics=compute_metrics,
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
