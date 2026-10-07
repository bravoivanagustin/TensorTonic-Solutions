import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    l = 1.0507
    a = 1.6733
    return [round(l*z, 4) if z > 0 else round(l*a*(math.e**z-1), 4) for z in x]