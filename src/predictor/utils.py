import numpy as np


def shuffle(dataset, random_seed=26):
    np.random.seed(random_seed)
    index = np.array([i for i in range(len(dataset))])
    np.random.shuffle(index)
    shuffle_dataset = []
    for i in index:
        shuffle_dataset.append(dataset[i])
    return np.array(shuffle_dataset)
