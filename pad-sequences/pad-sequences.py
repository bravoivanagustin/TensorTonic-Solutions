import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        length = max([len(seq) for seq in seqs]) if len(seqs) > 0 else 0
    else:
        length = max_len

    pad_seqs = np.zeros((len(seqs), length))
    
    for i in range(len(pad_seqs)):
        for j in range(length):
            if j < len(seqs[i]):
                pad_seqs[i, j] = seqs[i][j]
            else:
                pad_seqs[i, j] = pad_value

    pad_seqs = np.array(pad_seqs).astype(int)
    return pad_seqs