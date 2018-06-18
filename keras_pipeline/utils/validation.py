""" Script used for frequently used variable verifiers """


def is_int_like(x):
    return np.issubdtype(type(x), np.integer)


def is_float_like(x):
    return np.issubdtype(type(x), np.floating)


def is_list_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def is_valid_img_shape(x):
    return len(x) == 3


def is_valid_input_tensor(input_tensor):
    valid_type = type(input_tensor) == type(keras.Input(shape=(1, 1)))
    valid_len  = len(input_tensor) == 3

    return valid_type & valid_len
