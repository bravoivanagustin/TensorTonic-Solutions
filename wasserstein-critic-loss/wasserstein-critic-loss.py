import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    return sum(fake_scores) / len(fake_scores) - sum(real_scores) / len(real_scores)