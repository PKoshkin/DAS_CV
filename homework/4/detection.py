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


def get_data(img_dir, train_gt, shape):
    file_names = listdir(img_dir)
    source_images = [
        imread(join(img_dir, file_name))
        for file_name in file_names
    ]
    source_answers = [
        train_gt[file_name]
        for file_name in file_names
    ]
    resized_images = np.array([
        resize(image, shape, mode='reflect')
        for image in source_images
    ])
    resized_answers = np.array([[
            int(coord * shape[i % 2] / np.shape(image)[i % 2])
            for i, coord in enumerate(answer)
        ] for image, answer in zip(source_images, source_answers)
    ])
    return resized_images, resized_answers

def get_images(img_dir, shape):
    file_names = listdir(img_dir)
    source_images = [
        imread(join(img_dir, file_name))
        for file_name in file_names
    ]
    old_sizes = [
        np.shape(image)
        for image in source_images
    ]
    resized_images = np.array([
        resize(image, shape, mode='reflect')
        for image in source_images
    ])
    return resized_images, file_names, old_sizes

def build_model(model, input_shape, output_size):
    regularization_lambda = 5e-6

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
    def __init__(self, model, test_x, test_y):
        self._model = model

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
        print(model.evaluate(X_test, y_test, batch_size=128))

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

    train_x, train_y = get_data(train_img_dir, train_gt, img_size)
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
        optimizer=Adam(),
        metrics=['mse']
    )

    datagen = ImageDataGenerator()
    if validation:
        model.fit_generator(
            datagen.flow(train_x, train_y, batch_size=batch_size),
            steps_per_epoch=int(len(train_y) / batch_size),
            validation_data=(test_x, test_y),
            epochs=epochs,
            callbacks=[ValidationCallback(model, test_x, test_y)]
        )
    else:
        model.fit_generator(
            datagen.flow(train_x, train_y, batch_size=batch_size),
            steps_per_epoch=int(len(train_y) / batch_size),
            epochs=epochs
        )


def detect(model, test_img_dir):
    shape = IMG_SIZE
    test_x, file_names, old_sizes = get_images(test_img_dir, shape)
    answers = model.predict(test_x, batch_size=128)
    resized_back_answers = np.array([[
            int(coord * old_shape[i % 2] / shape[i % 2])
            for i, coord in enumerate(answer)
        ] for old_shape, answer in zip(old_sizes, answers)
    ])
    return {
        file_name: answer
        for file_name, answer in zip(file_names, resized_back_answers)
    }
