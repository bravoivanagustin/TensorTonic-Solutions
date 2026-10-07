import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    return [z if z > 0 else alpha*(math.e**z-1) for z in x] 