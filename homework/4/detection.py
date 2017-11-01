import numpy as np

from skimage.transform import resize
from skimage.io import imread

from sklearn.model_selection import train_test_split

from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam

from os.path import abspath, dirname, join, exists
from os import listdir


IMG_SIZE = (100, 100, 3)


class LabeledBatchIterator:
    def __init__(self, filenames, directory, resize_shape, batch_size, train_gt):
        self._filenames = filenames
        self._directory = directory
        self._resize_shape = resize_shape
        self._batch_size = batch_size
        self._train_gt = train_gt

    def _get_image(self, index):
        return imread(join(self._directory, self._filenames[index]))

    def __iter__(self):
        return self

    def __next__(self):
        random_indexes = np.random.permutation(len(self._filenames))[:self._batch_size]
        return np.array([
            resize(self._get_image(index), self._resize_shape, mode='reflect')
            for index in random_indexes
        ]), np.array([[
                int(coord * self._resize_shape[i % 2] / np.shape(self._get_image(index))[i % 2])
                for i, coord in enumerate(self._train_gt[self._filenames[index]])
            ] for index in random_indexes
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
    label_len = 28
    img_size = IMG_SIZE
    code_dir = dirname(abspath(__file__))
    model_path = join(code_dir, 'facepoints_model.hdf5')

    all_filenames = listdir(train_img_dir)
    np.random.shuffle(all_filenames)
    if validation:
        train_filenames = all_filenames[int(len(train_filenames) * validation):]
        test_filenames = all_filenames[:int(len(train_filenames) * validation)]

    if fast_train or not exists(model_path):
        model = Sequential()
        build_model(model, img_size, label_len)
    else:
        model = load_model(model_path)

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(decay=1e-3),
        metrics=['mse']
    )

    if validation:
        model.fit_generator(
            LabeledBatchIterator(train_filenames, train_img_dir, img_size, batch_size, train_gt),
            steps_per_epoch=int(len(train_filenames) / batch_size),
            epochs=epochs,
            callbacks=[ValidationCallback(
                model,
                LabeledBatchIterator(test_filenames, train_img_dir, img_size, batch_size, train_gt),
                int(len(test_filenames) / batch_size)
            )]
        )
    else:
        model.fit_generator(
            LabeledBatchIterator(all_filenames, train_img_dir, img_size, batch_size, train_gt),
            steps_per_epoch=int(len(all_filenames) / batch_size),
            epochs=epochs
        )


def detect(model, test_img_dir):
    img_size = IMG_SIZE
    filenames = listdir(test_img_dir)
    batch_size = 128
    random_indexes = np.random.permutation(len(filenames))[:batch_size]
    filenames = [filenames[index] for index in random_indexes]
    test_images = np.array([
        resize(imread(join(test_img_dir, filename)), img_size, mode='reflect')
        for filename in filenames
    ])
    old_shapes = np.array([
        np.shape(imread(join(test_img_dir, filename)))
        for filename in filenames
    ])
    answers = model.predict(test_images, batch_size=batch_size)
    resized_back_answers = np.array([[
            int(coord * old_shape[i % 2] / img_size[i % 2])
            for i, coord in enumerate(answer)
        ] for old_shape, answer in zip(old_shapes, answers)
    ])
    return {
        filename: answer
        for filename, answer in zip(filenames, resized_back_answers)
    }
