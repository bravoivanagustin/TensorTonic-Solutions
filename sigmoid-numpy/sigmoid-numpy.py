import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    def isscalar(x):
        return isinstance(x, float) or isinstance(x, int)
    
    if isscalar(x):
        return 1/(1+np.exp(-float(x)))
    else:
        return np.array([sigmoid(z) for z in x])