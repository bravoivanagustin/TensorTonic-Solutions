import numpy as np

def relu(x) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    def is_scalar(x):
        return isinstance(x, float) or isinstance(x, int)

    if is_scalar(x):
        return max([np.array(0),np.array(x)])
    else:
        return np.array([relu(z) for z in x])