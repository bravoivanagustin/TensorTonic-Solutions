import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    pos_enc = np.array([[np.sin(pos/(base**(i/d_model))) if i % 2 == 0 else np.cos(pos/(base**((i-1)/d_model))) for i in range(d_model)] for pos in range(seq_len)])
    return pos_enc