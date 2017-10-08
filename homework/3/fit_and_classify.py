import numpy as np
from collections import Counter

def fit_and_classify(train_features, train_labels, test_features):
    pass


def extract_hog(image):
    def get_brightness(image):
        return 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]

    def get_derivative(brightness_matrix, axis):
        if axis != 0:
            brightness_matrix = brightness_matrix.transpose()
        derivative_matrix = np.concatenate((
            [brightness_matrix[1,:] - brightness_matrix[0,:]],
            brightness_matrix[2:,:] - brightness_matrix[:-2,:],
            [brightness_matrix[(np.shape(brightness_matrix)[0] - 1),:] - brightness_matrix[(np.shape(brightness_matrix)[0] - 2),:]]
        ), axis=0)
        if axis != 0:
            derivative_matrix = derivative_matrix.transpose()
        return derivative_matrix

    def get_gradient_normal(brightness_matrix):
        return np.sqrt(get_derivative(brightness_matrix, 0) ** 2 + get_derivative(brightness_matrix, 1) ** 2)

    def get_gradient_direction(brightness_matrix):
        return np.arctan2(get_derivative(brightness_matrix, 0), get_derivative(brightness_matrix, 1))

    CELL_ROWS = 100
    CELL_COLUMNS = 100
    BIN_COUNT = 8

    brightness_matrix = get_brightness(image)
    gradient_normal_matrix = get_gradient_normal(brightness_matrix)
    gradient_direction_matrix = get_gradient_direction(brightness_matrix)

    angle_splits = np.linspace(-np.pi, np.pi, BIN_COUNT + 1)
    bar_chart = Counter()
    for y in range(np.shape(brightness_matrix)[0]):
        for x in range(np.shape(brightness_matrix)[1]):
            bar_chart[(gradient_direction_matrix[y][x] + np.pi) / (2 * np.pi) * BIN_COUNT] += gradient_normal_matrix[y][x]
    return bar_chart
