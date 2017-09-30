import numpy as np


def get_brightness_matrix(image):
    return 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]


def get_location(coordinate, size):
    return np.arange(max(0, coordinate - 1), min(coordinate + 2, size))


def get_derivative(brightness_matrix, axis):
    if axis != 0:
        brightness_matrix = brightness_matrix.transpose()
    result = np.concatenate((
        [brightness_matrix[1,:] - brightness_matrix[0,:]],
        brightness_matrix[2:,:] - brightness_matrix[:-2,:],
        [brightness_matrix[(np.shape(brightness_matrix)[0] - 1),:] - brightness_matrix[(np.shape(brightness_matrix)[0] - 2),:]]
    ), axis=0)
    if axis != 0:
        result = result.transpose()
    return result


def get_gradient_normal_matrix(brightness_matrix, mask):
    result = np.sqrt(get_derivative(brightness_matrix, 0) ** 2 + get_derivative(brightness_matrix, 1) ** 2)
    if mask is not None:
        result += np.shape(mask)[0] * np.shape(mask)[1] * mask
    return result


def get_seams_energy_matrix(gradient_normal_matrix):
    shape = np.shape(gradient_normal_matrix)
    seams_energy_matrix = np.zeros_like(gradient_normal_matrix)
    seams_energy_matrix[0] += gradient_normal_matrix[0]
    for y in range(1, shape[0]):
        seams_energy_matrix[y] += np.array([
            gradient_normal_matrix[y, x] +
            np.min(seams_energy_matrix[(y - 1), get_location(x, shape[1])])
            for x in range(shape[1])
        ])
    return seams_energy_matrix


def get_vertical_seam(seams_energy_matrix):
    shape = np.shape(seams_energy_matrix)
    seam = np.zeros_like(seams_energy_matrix)
    answer = np.argmin(seams_energy_matrix[-1])
    seam[shape[0] - 1, answer] += 1
    for y in range(shape[0] - 2, -1, -1):
        min_index = np.argmin(seams_energy_matrix[y, get_location(answer, shape[1])]) - 1
        if answer == 0:
            min_index += 1
        answer += min_index
        seam[y, answer] += 1
    return seam


def seam_carve(image, mode, mask=None):
    brightness_matrix = get_brightness_matrix(image)
    gradient_normal_matrix = get_gradient_normal_matrix(brightness_matrix, mask)
    if mode.split(' ')[0] != 'horizontal':
        gradient_normal_matrix = gradient_normal_matrix.transpose()
    seams_energy_matrix = get_seams_energy_matrix(gradient_normal_matrix)
    seam = get_vertical_seam(seams_energy_matrix)
    if mode.split(' ')[0] != 'horizontal':
        seam = seam.transpose()
    return None, None, seam
