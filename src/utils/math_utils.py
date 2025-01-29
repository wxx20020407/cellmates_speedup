import numpy as np


def l_from_p(p_change, n_states=4) -> float | np.ndarray:
    """
    Inverse formula to determine length from p_change (probability of changing *to any state*).
    Parameters
    ----------
    p_change: float or np.ndarray, probability of changing to any state
    n_states: int, number of copy number states

    Returns
    -------
    float, length parameter
    """
    return - 1 / n_states * np.log(1 - n_states / (n_states - 1) * p_change)

def p_from_l(l, n_states=4) -> float | np.ndarray:
    """
    Formula to determine p_change (probability of changing *to any state*) from length.
    Parameters
    ----------
    l: float or np.ndarray, length parameter
    n_states: int, number of copy number states

    Returns
    -------
    float, probability of changing to any state
    """
    return (1 - np.exp(-n_states * l)) * (n_states - 1) / n_states

