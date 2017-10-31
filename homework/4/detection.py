import numpy as np

from skimage.transform import resize
from skimage.io import imread

from sklearn.model_selection import train_test_split

import threading

from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam

from os.path import abspath, dirname, join, exists
from os import listdir


IMG_SIZE = (100, 100, 3)


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


def get_images(img_dir):
    file_names = listdir(img_dir)
    return np.array([
        imread(join(img_dir, file_name))
        for file_name in file_names
    ])

def get_answers(img_dir, train_gt):
    file_names = listdir(img_dir)
    return np.array([
        train_gt[file_name]
        for file_name in file_names
    ])

def build_model(model, input_shape, output_size):
    regularization_lambda = 5e-3

    def add_Conv2D_relu(model, n_filter, filters_size, input_shape=None):
        if input_shape is not None:
            model.add(Conv2D(
                n_filter, filters_size, padding='same',
                kernel_regularizer=regularizers.l2(regularization_lambda),
                activation='elu', input_shape=input_shape,
            ))
        else:
            model.add(Conv2D(
                n_filter, filters_size, padding='same',
                kernel_regularizer=regularizers.l2(regularization_lambda),
                activation='elu'
            ))


    add_Conv2D_relu(model, 64, (3, 3), input_shape)
    add_Conv2D_relu(model, 64, (3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    add_Conv2D_relu(model, 128, (3, 3))
    add_Conv2D_relu(model, 128, (3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    add_Conv2D_relu(model, 256, (3, 3))
    add_Conv2D_relu(model, 256, (3, 3))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='elu', kernel_regularizer=regularizers.l2(regularization_lambda)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='elu', kernel_regularizer=regularizers.l2(regularization_lambda)))

    model.add(Dense(output_size))

class ValidationCallback:
    def __init__(self, model, generator, steps):
        self._model = model
        self._generator = generator
        self._steps = steps

    def set_model(self, model):
        pass

    def set_params(self, params):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        print(model.evaluate_generator(self._generator, self._steps))

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

def train_detector(train_gt, train_img_dir, fast_train, validation=0.0):
    epochs = 1 if fast_train else 1000
    batch_size = 32
    img_size = IMG_SIZE
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'facepoints_model.hdf5')

    train_x = get_images(train_img_dir)
    train_y = get_answers(train_img_dir, train_gt)
    if validation:
        train_x, test_x, train_y, test_y = train_test_split(
            resized_x, resized_y, test_size=0.3, random_state=42
        )

    if fast_train or not exists(model_path):
        model = Sequential()
        build_model(model, img_size, len(train_y[0]))
    else:
        model = load_model(model_path)

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(decay=1e-3),
        metrics=['mse']
    )

    datagen = ResizeImageGenerator(img_size)
    if validation:
        model.fit_generator(
            datagen.flow(train_x, train_y, batch_size),
            steps_per_epoch=int(len(train_y) / batch_size),
            epochs=epochs,
            callbacks=[ValidationCallback(
                model,
                datagen.flow(test_x, test_y, batch_size),
                int(len(test_y) / batch_size),
            )]
        )
    else:
        model.fit_generator(
            datagen.flow(train_x, train_y, batch_size),
            steps_per_epoch=int(len(train_y) / batch_size),
            epochs=epochs
        )


def detect(model, test_img_dir):
    shape = IMG_SIZE
    test_x = get_images(test_img_dir)
    resized_test_x = np.array([
        resize(x, shape, mode='reflect')
        for x in test_x
    ])
    answers = model.predict(resized_test_x, batch_size=128)
    resized_back_answers = np.array([[
            int(coord * np.shape(image)[i % 2] / shape[i % 2])
            for i, coord in enumerate(answer)
        ] for image, answer in zip(test_x, answers)
    ])
    return {
        file_name: answer
        for file_name, answer in zip(file_names, resized_back_answers)
    }
