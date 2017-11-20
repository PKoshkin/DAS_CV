import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread

IMG_SIZE = (299, 299, 3)
NUM_CLASSES = 50


class BatchIterator:
    def __init__(self, filenames, directory, resize_shape, batch_size, train_gt):
        self._filenames = filenames
        self._directory = directory
        self._resize_shape = resize_shape
        self._batch_size = batch_size
        self._train_gt = train_gt

    def _get_image(self, index):
        return imread(os.path.join(self._directory, self._filenames[index]))

    def __iter__(self):
        return self

    def __next__(self):
        random_indexes = np.random.permutation(len(self._filenames))[:self._batch_size]
        batch_filenames = [self._filenames[index] for index in random_indexes]
        labels = np.zeros((self._batch_size, NUM_CLASSES))
        labels_indexes = np.array([
            self._train_gt[filename]
            for filename in batch_filenames
        ])
        labels[np.arange(self._batch_size), labels_indexes] = 1
        return np.array([
            resize(self._get_image(index), self._resize_shape, mode='reflect')
            for index in random_indexes
        ]), labels


class VerboseClallback:
    def __init__(self, model, directory):
        self._model = model
        self._directory = directory
        self._counter = 1

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
        print(
            str(self._counter) +
            ') val_categorical_accuracy: '
            + str(logs['val_categorical_accuracy'])
        )
        self._counter += 1

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass


def get_model(file_name='birds_model.hdf5', image_shape=IMG_SIZE, regularization_lambda=1e-3):
    from keras import regularizers
    from keras.layers import Dense, Dropout
    from keras.models import Model
    new_model = not os.path.exists(file_name)
    if new_model:
        from keras.applications.xception import Xception
        initial_model = Xception(
            include_top=False, weights='imagenet',
            input_shape=image_shape, pooling='avg'
        )
        last = initial_model.output

        nn = Dense(
            1024, activation='elu',
            kernel_regularizer=regularizers.l2(regularization_lambda)
        )(last)
        nn = Dense(
            1024, activation='elu',
            kernel_regularizer=regularizers.l2(regularization_lambda)
        )(nn)

        prediction = Dense(
            NUM_CLASSES, activation='softmax',
            kernel_regularizer=regularizers.l2(regularization_lambda)
        )(nn)

        model = Model(initial_model.input, prediction)

        for layer in initial_model.layers:
            layer.trainable = False
    else:
        from keras.models import load_model
        model = load_model(file_name)
    return model


def train_classifier(train_gt, train_img_dir, fast_train, validation=0.3):
    from keras.optimizers import Adam
    from sklearn.model_selection import train_test_split
    model = get_model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-4, decay=1e-2),
        metrics=['categorical_accuracy']
    )
    image_filenames = os.listdir(train_img_dir)
    batch_size = 4
    epochs = 1 if fast_train else 60
    steps_per_epoch = 4 if fast_train else int(len(image_filenames) / batch_size)
    model.fit_generator(
        BatchIterator(image_filenames, train_img_dir, IMG_SIZE, batch_size, train_gt),
        steps_per_epoch=steps_per_epoch, epochs=epochs
    )

    return model

def classify(model, test_img_dir):
    result = {}
    img_size=IMG_SIZE
    batch_size = 8
    filenames = os.listdir(test_img_dir)
    for begin_index in range(0, len(filenames), batch_size):
        current_filenames = [
            filenames[index] for index in range(begin_index, min(begin_index + batch_size, len(filenames)))
        ]
        test_images = np.array([
            resize(imread(os.path.join(test_img_dir, filename)), img_size, mode='reflect')
            for filename in current_filenames
        ])
        answers = model.predict(test_images, batch_size=batch_size)
        for filename, answer in zip(current_filenames, answers):
            result[filename] = np.argmax(answer)
    return result