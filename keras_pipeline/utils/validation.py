""" Script used for frequently used variable verifiers """


def is_int_like(x):
    return np.issubdtype(type(x), np.integer)


def is_float_like(x):
    return np.issubdtype(type(x), np.floating)


def is_list_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def is_positive(x):
    return x > 0
