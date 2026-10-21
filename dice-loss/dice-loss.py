import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    p_aux, y_aux = np.array(p).flatten(), np.array(y).flatten()
    dice = 2*sum([p_aux[i]*y_aux[i] for i in range(len(p_aux))]) + eps
    dice = dice / (sum(p_aux) + sum(y_aux) + eps)
    return float(1-dice)