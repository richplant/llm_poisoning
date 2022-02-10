import numpy as np
from datasets import load_dataset
from pathlib import Path

SEED = 42
FLIP_RATES = [
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5
]
RND = np.random.default_rng(SEED)
GLUE = [
    "cola",
    "mrpc",
    "sst2",
    "wnli",
    "rte",
]


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


def main():
    output_dir = Path('./datasets')
    output_dir.mkdir(parents=True, exist_ok=True)

    # export glue
    for task_name in GLUE:
        data = load_dataset('glue', task_name)['train']
        for flip_ratio in FLIP_RATES:
            data_flip = data.map(perturb_binary_labels, fn_kwargs={'thold': flip_ratio})
            output_dir.joinpath(task_name).mkdir(exist_ok=True)
            csv_path = output_dir.joinpath(task_name).joinpath(f'{task_name}_{flip_ratio}.csv')
            data_flip.to_csv(csv_path)

    # export ag_news
    data = load_dataset('ag_news')['train']
    for flip_ratio in FLIP_RATES:
        data_flip = data.map(perturb_labels, fn_kwargs={'thold': flip_ratio})
        output_dir.joinpath('ag_news').mkdir(exist_ok=True)
        csv_path = output_dir.joinpath('ag_news').joinpath(f'ag_news_{flip_ratio}.csv')
        data_flip.to_csv(csv_path)


if __name__ == '__main__':
    main()
