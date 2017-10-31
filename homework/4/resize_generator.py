import numpy as np
from skimage.transform import resize

class ResizeImageGenerator:
    def __init__(self, shape):
        self._shape = shape

    @staticmethod
    def resize(image, answer, shape):
        return (
            np.array(resize(image, shape, mode='reflect')),
            np.array([
                int(coord * shape[i % 2] / np.shape(image)[i % 2])
                for i, coord in enumerate(answer)
            ])
        )

    def _resize(self, image, answer):
        return ResizeImageGenerator.resize(image, answer, self._shape)

    def flow(self, x, y, batch_size=32):
        return BatchIterator(x, y, self, batch_size)


class BatchIterator:
    def __init__(self, x, y, image_generator, batch_size):
        self._x = x
        self._y = y
        self._image_generator = image_generator
        self._batch_size = batch_size

    def _get_batch(self):
        random_indexes = np.random.permutation(len(self._x))[:self._batch_size]
        return self._x[random_indexes], self._y[random_indexes]
    
    def _resize(self, x, y):
        return self._image_generator._resize(x, y)

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self._get_batch()
        x_batch, y_batch = [], []
        for current_x, current_y in zip(x, y):
            resized_x, resized_y = self._resize(current_x, current_y)
            x_batch.append(resized_x)
            y_batch.append(resized_y)
        return np.array(x_batch), np.array(y_batch)
