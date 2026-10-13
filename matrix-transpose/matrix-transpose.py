import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A_t = np.array([[A[j][i] for j in range(len(A))] for i in range(len(A[0]))])
    return A_t