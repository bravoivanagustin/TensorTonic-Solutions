import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    n = len(m)

    m_t = [beta1 * m[i] + (1 - beta1) * grad[i] for i in range(n)]
    v_t = [beta2 * v[i] + (1 - beta2) * grad[i]**2 for i in range(n)]
    
    m_t_h = [m_t[i] / (1 - beta1**t) for i in range(n)]
    v_t_h = [v_t[i] / (1 - beta2**t) for i in range(n)]

    theta_t = [param[i] - lr * m_t_h[i] / (np.sqrt(v_t_h[i]) + eps) for i in range(n)]

    return theta_t, m_t, v_t