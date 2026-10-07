import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    def is_scalar(x):
        return isinstance(x, float) or isinstance(x, int)

    if is_scalar(x):
        return np.array(x/(1+np.exp(-x)))
    else:
        return np.array([swish(z) for z in x])