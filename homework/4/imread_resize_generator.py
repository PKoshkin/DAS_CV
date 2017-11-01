import numpy as np
from skimage.transform import resize
from skimage.io import imread

class BatchIterator:
    def __init__(self, file_names, resize_shape, batch_size, train_gt):
        self._file_names = file_names
        self._resize_shape = resize_shape
        self._batch_size = batch_size
        self._train_gt = train_gt

    def __iter__(self):
        return self

    def __next__(self):
        random_indexes = np.random.permutation(len(self._file_names))[:self._batch_size]
        return np.array([
            resize(imread(self._file_names[index]), self._resize_shape, mode='reflect')
            for index in random_indexes
        ]), np.array([[
                int(coord * self._resize_shape[i % 2] / np.shape(imread(self._file_names[index]))[i % 2])
                for i, coord in enumerate(self._train_gt[self._file_names[index]])
            ] for index in random_indexes
        ])
