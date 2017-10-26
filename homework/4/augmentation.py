import numpy as np

class ShiftImageException(Exception):
    pass

class Augmentation:
    '''
    rotation, featurewise_center and featurewise_std_normalization may be added futher
    '''
    def __init__(self, width_shift=0, height_shift=0,
                 horizontal_flip=False, vertical_flip=False):
        self._width_shift = width_shift
        self._height_shift = height_shift
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip

    @staticmethod
    def _shift_image(image, shift, axis):
        height, width = np.shape(image)[:2]
        if shift > 0:
            edge_coordinate = -1
        else:
            edge_coordinate = 0
        if axis == 0:
            line = np.reshape(image[edge_coordinate], (1, width, 3))
            if shift > 0:
                crop = image[shift:]
            else:
                crop = image[:shift]
        elif axis == 1:
            line = np.reshape(image[:,edge_coordinate], (height, 1, 3))
            if shift > 0:
                crop = image[:,shift:]
            else:
                crop = image[:,:shift]
        else:
            raise ShiftImageException()
        edge = np.concatenate(tuple(line for i in range(abs(shift))), axis=axis)
        if shift > 0:
            return np.concatenate((crop, edge), axis=axis)
        else:
            return np.concatenate((edge, crop), axis=axis) 

    def apply(self, image, coordinates):
        image = np.copy(image)
        coordinates = np.copy(coordinates)

        if self._horizontal_flip:
            image = image[::,::-1]
            coordinates[0::2] = np.shape(image)[1] - coordinates[0::2]

        if self._vertical_flip:
            image = image[::-1]
            coordinates[1::2] = np.shape(image)[0] - coordinates[1::2]

        if self._width_shift != 0:
            image = Augmentation._shift_image(image, self._width_shift, 1)
            coordinates[0::2] -= self._width_shift

        if self._height_shift != 0:
            image = Augmentation._shift_image(image, self._height_shift, 0)
            coordinates[1::2] -= self._height_shift

        return image, coordinates

    
def get_random_augmentation(width_shift=0, height_shift=0,
                            horizontal_flip=False, vertical_flip=False):
    return Augmentation(
        np.random.randint(-width_shift, width_shift),
        np.random.randint(-height_shift, height_shift),
        bool(np.random.binomial(1, 0.5)) if horizontal_flip else False,
        bool(np.random.binomial(1, 0.5)) if vertical_flip else False
    )


class ImageGenerator:
    def __init__(self, width_shift=5, height_shift=5,
                 horizontal_flip=True, vertical_flip=False):
        self._width_shift = width_shift
        self._height_shift = height_shift
        self._horizontal_flip = horizontal_flip
        self._vertical_flip = vertical_flip

    def random_transform(self, x, y):
        random_augmentation = get_random_augmentation(
            self._width_shift,
            self._height_shift,
            self._horizontal_flip,
            self._vertical_flip
        )
        return random_augmentation.apply(x, y)

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
    
    def _randomize(self, x, y):
        return self._image_generator.random_transform(x, y)

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self._get_batch()
        x_batch, y_batch = [], []
        for current_x, current_y in zip(x, y):
            random_x, random_y = self._randomize(current_x, current_y)
            x_batch.append(random_x)
            y_batch.append(random_y)
        return np.array(x_batch), np.array(y_batch)
