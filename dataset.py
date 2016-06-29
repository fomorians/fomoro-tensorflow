import os
import pickle
import hashlib
import requests

import numpy as np

from sklearn.cross_validation import train_test_split

SEED = 67

NUM_SECONDS = 15
WIDTH, HEIGHT, DEPTH = 24, 24, 3
TEST_SPLIT = 0.2
VALID_SPLIT = 0.5

def download_file(url, dest):
    r = requests.get(url, stream=True)
    with open(dest, 'wb+') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk:
                f.write(chunk)
    return dest

def md5_hash(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class Dataset(object):

    def __init__(self, images, labels):
        images = images[..., :1]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch

        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)

            self._images = self._images[perm]
            self._labels = self._labels[perm]

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

class Datasets(object):

    def __init__(self, train, validation, test):
        self.train = train
        self.validation = validation
        self.test = test

def read_data_sets(dataset_destination, dataset_source, dataset_hash=None):
    if not os.path.exists(dataset_destination):
        print('Downloading dataset...')
        download_file(dataset_source, dataset_destination)

    if dataset_hash:
        print('Checking dataset hash...')
        assert md5_hash(dataset_destination) == dataset_hash

    with open(dataset_destination, 'rb') as f:
        print('Loading dataset...')
        X, yh, ym, ys = pickle.load(f)

    X_train, X_test, yh_train, yh_test = \
        train_test_split(X, yh, test_size=TEST_SPLIT, random_state=SEED)

    X_valid, X_test, yh_valid, yh_test = \
        train_test_split(X_test, yh_test, test_size=VALID_SPLIT, random_state=SEED)

    train = Dataset(X_train, yh_train)
    validation = Dataset(X_valid, yh_valid)
    test = Dataset(X_test, yh_test)

    return Datasets(train=train, validation=validation, test=test)
