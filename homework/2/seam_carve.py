import numpy as np


def get_brightness_matrix(image):
    return np.array([[
            0.299 * image[y][x][0] + 0.587 * image[y][x][1] + 0.114 * image[y][x][2]
            for x in range(np.shape(image)[1])
        ] for y in range(np.shape(image)[0])
    ])

def get_upper_index(coordinate, size):
    return (coordinate + 1) if (coordinate + 1) != size else coordinate


def get_lower_index(coordinate):
    return (coordinate - 1) if coordinate != 0 else coordinate


def get_y_difference(brightness_matrix, x, y):
    return (
        brightness_matrix[get_upper_index(y, np.shape(brightness_matrix)[0])][x] -
        brightness_matrix[get_lower_index(y)][x]
    )


def get_derivative(brightness_matrix, axis):
    if axis != 0:
        brightness_matrix = brightness_matrix.transpose()
    result = np.array([[
            get_y_difference(brightness_matrix, x, y)
            for x in range(np.shape(brightness_matrix)[1])
        ] for y in range(np.shape(brightness_matrix)[0])
    ])
    if axis != 0:
        brightness_matrix = brightness_matrix.transpose()
        result = result.transpose()
    return result


def get_gradient_normal_matrix(brightness_matrix):
    y_derivative = get_derivative(brightness_matrix, 0)
    x_derivative = get_derivative(brightness_matrix, 1)
    return np.array([[
            np.sqrt(x_derivative[y][x] ** 2 + y_derivative[y][x] ** 2)
            for x in range(np.shape(brightness_matrix)[1])
        ] for y in range(np.shape(brightness_matrix)[0])
    ])


def get_least_energy_continuation(gradient_normal_matrix, x, y):
    return gradient_normal_matrix[y][x] + min(
        gradient_normal_matrix[y - 1][get_upper_index(x, np.shape(gradient_normal_matrix)[1])],
        gradient_normal_matrix[y - 1][x],
        gradient_normal_matrix[y - 1][get_lower_index(x)],
    )


def get_vertical_seam(gradient_normal_matrix):
    shape = np.shape(gradient_normal_matrix)
    temp_matrix = [gradient_normal_matrix[0]]
    for y in range(shape[0] - 2, -1, -1):
        new_line = [
            get_least_energy_continuation(gradient_normal_matrix, x, y)
            for x in range(shape[1])
        ]
    answers = [np.argmin(temp_matrix[-1])]
    for y in range(1, shape[0]):
        answers.append(answers[-1] + np.argmin(
                gradient_normal_matrix[y][get_lower_index(answers[-1]):(get_upper_index(answers[-1], shape[1]) + 1)]
            ) + (1 if answers[-1] == 0 else 0) - 1
        )
    return answers


def get_cuted_verticaly_by_seam(image, seam):
    return np.array([
        [x for x in image[y][:seam[y]]] + [x for x in image[y][(seam[y] + 1):]]
        for y in range(np.shape(image)[0])
    ])


def get_cuted(image, axis):
    if axis != 0:
        image = np.swapaxes(image, 0, 1)
    brightness_matrix = get_brightness_matrix(image)
    gradient_normal_matrix = get_gradient_normal_matrix(brightness_matrix)
    seam = get_vertical_seam(gradient_normal_matrix)
    result = get_cuted_verticaly_by_seam(image, seam)
    if axis != 0:
        result = np.swapaxes(result, 0, 1)
        image = np.swapaxes(image, 0, 1)
    return result
