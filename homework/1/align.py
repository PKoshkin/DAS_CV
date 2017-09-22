import numpy as np

MAX_SIZE = 500


def get_channels(image):
    height = int(np.shape(image)[0] / 3)
    return (
        image[(height * 2):(height * 3)],
        image[height:(height * 2)],
        image[:height]
    )


def clipped(image, height_border, width_border):
    return image[height_border:-height_border, width_border:-width_border]


def get_clipped_channels(image, clipped_percent=5):
    channels = get_channels(image)
    return tuple(
        clipped(
            channel,
            int(np.shape(channel)[0] * 0.01 * clipped_percent),
            int(np.shape(channel)[1] * 0.01 * clipped_percent)
        )
        for channel in channels
    )


def get_shifted(image, x_shift=0, y_shift=0):
    image = image[y_shift:] if y_shift >= 0 else image[:y_shift]
    image = image[:, x_shift:] if x_shift >= 0 else image[:, :x_shift]
    return image


def MSE(image_1, image_2):
    return np.mean((image_1 - image_2) ** 2)


def get_best_shift(static_channel, shifting_channel, max_shift=15):
    best_MSE = MSE(static_channel, shifting_channel)
    best_x, best_y = 0, 0
    for x_shift in range(-max_shift, max_shift + 1):
        for y_shift in range(-max_shift, max_shift + 1):
            new_MSE = MSE(
                get_shifted(shifting_channel, x_shift, y_shift),
                get_shifted(static_channel, -x_shift, -y_shift)
            )
            if new_MSE < best_MSE:
                best_x = x_shift
                best_y = y_shift
                best_MSE = new_MSE
    return best_x, best_y


def get_compresed(channel):
    return channel[::2,::2]


def get_fast_best_shift(static_channel, shifting_channel):
    pyramid = [(static_channel, shifting_channel)]
    while (np.shape(static_channel)[0] > MAX_SIZE) or (np.shape(shifting_channel)[1] > MAX_SIZE):
        static_channel = get_compresed(static_channel)
        shifting_channel = get_compresed(shifting_channel)
        pyramid.append((static_channel, shifting_channel))

    x_shift, y_shift = get_best_shift(pyramid[-1][0], pyramid[-1][1], 30)
    for compresed_static_channel, compresed_shifting_channel in reversed(pyramid[:-1]):
        x_shift *= 2
        y_shift *= 2
        tmp_x_shift, tmp_y_shift = get_best_shift(
            get_shifted(compresed_static_channel, -x_shift, -y_shift),
            get_shifted(compresed_shifting_channel, x_shift, y_shift),
            1
        )
        x_shift += tmp_x_shift
        y_shift += tmp_y_shift

    return x_shift, y_shift


def align(image, g_coord):
    channels = get_clipped_channels(image)
    height = int(np.shape(channels)[1] * 10 / 9)
    x_shift_red, y_shift_red = get_fast_best_shift(channels[1], channels[0])
    channels = [
        get_shifted(channels[0], -x_shift_red, -y_shift_red),
        get_shifted(channels[1], x_shift_red, y_shift_red),
        get_shifted(channels[2], x_shift_red, y_shift_red)
    ]
    x_shift_blue, y_shift_blue = get_fast_best_shift(channels[1], channels[2])
    channels = [
        get_shifted(channels[0], x_shift_blue, y_shift_blue),
        get_shifted(channels[1], x_shift_blue, y_shift_blue),
        get_shifted(channels[2], -x_shift_blue, -y_shift_blue)
    ]
    channels = np.moveaxis(np.array(channels), 0, 2)
    return [
        channels,
        (g_coord[0] + y_shift_blue - height, g_coord[1] + x_shift_blue),
        (g_coord[0] + y_shift_red + height, g_coord[1] + x_shift_red)
    ]
