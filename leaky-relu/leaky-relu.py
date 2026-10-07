import numpy as np

def leaky_relu(x: list | float, alpha: float) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    def is_scalar(x):
        return isinstance(x, float) or isinstance(x, int)

    if is_scalar(x):
        return np.array(alpha*x) if x < 0 else np.array(x)
    else:
        return np.array([leaky_relu(z, alpha) for z in x])
    pass