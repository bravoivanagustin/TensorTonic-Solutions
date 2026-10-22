import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    if len(y) == 0:
        return float(0)
    else:
        prob = [y.count(x)/len(y) for x in np.unique(y)]
        return sum([-p*np.log2(p) for p in prob])