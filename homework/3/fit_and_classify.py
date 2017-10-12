import numpy as np
from math import floor
from collections import Counter
from sklearn.svm import SVC

def fit_and_classify(train_features, train_labels, test_features):
    model = SVC(kernel='linear')
    model.fit(train_features, train_labels)
    return model.predict(test_features)


def extract_hog(image, cell_rows=8, cell_columns=8, bin_count=11, epsilon=1e-50, block_row_cells=2, block_collumn_cells=2, crop_pixels_propotion=0.3):
    crop_row_pixels_number = int(np.shape(image)[0] * crop_pixels_propotion)
    crop_column_pixels_number = int(np.shape(image)[1] * crop_pixels_propotion)
    image = image[crop_row_pixels_number:-crop_row_pixels_number, crop_column_pixels_number:-crop_column_pixels_number,:]

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

    def get_bar_chart(gradient_normal_matrix, gradient_direction_matrix):
        bar_chart = Counter({i: 0.0 for i in range(bin_count)})
        for y in range(np.shape(gradient_normal_matrix)[0]):
            for x in range(np.shape(gradient_normal_matrix)[1]):
                bar_chart[min(floor((gradient_direction_matrix[y][x] + np.pi) / (2 * np.pi) * bin_count), 7)] += gradient_normal_matrix[y][x]
        return bar_chart

    def get_cells_slices(shape):
        x_linspace = (np.linspace(0, 1, cell_rows + 1) * shape[0]).astype(int)
        y_linspace = (np.linspace(0, 1, cell_columns + 1) * shape[1]).astype(int)
        return [[
                (slice(y, y_next), slice(x, x_next))
                for x, x_next in zip(x_linspace, x_linspace[1:])
            ] for y, y_next in zip(y_linspace, y_linspace[1:])
        ]

    def get_block_features(cells_bar_charts):
        result = np.array([
            bar_chart[key]
            for bar_chart in cells_bar_charts
            for key in bar_chart
        ])
        result /= np.sqrt(np.sum(result ** 2) + epsilon)
        return result

    brightness_matrix = get_brightness(image)
    gradient_normal_matrix = get_gradient_normal(brightness_matrix)
    gradient_direction_matrix = get_gradient_direction(brightness_matrix)
    slices = get_cells_slices(np.shape(brightness_matrix))
    cells_bar_charts = np.array([[
            get_bar_chart(gradient_normal_matrix[current_slice], gradient_direction_matrix[current_slice])
            for current_slice in slices_row
        ] for slices_row in slices
    ])

    return np.concatenate(tuple([
        get_block_features(cells_bar_charts[i:(i + block_row_cells), j:(j + block_collumn_cells)].flatten())
        for j in range(np.shape(cells_bar_charts)[1] - block_collumn_cells + 1)
        for i in range(np.shape(cells_bar_charts)[0] - block_row_cells + 1)
    ]), axis=0)
