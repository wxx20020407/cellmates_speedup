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

def compute_cn_changes(cn_profile: np.ndarray, pairs: list = None) -> list:
    """
    Compute the changes in copy number states for a given profile according to the CopyTree model.
    Parameters
    ----------
    cn_profile: np.ndarray, shape (n_nodes, n_sites), copy number profiles for each node
    pairs: list of tuples, pairs of nodes to compare (default: [(0, 1)])

    Returns
    -------
    list of changes for each pair of nodes
    """
    if pairs is None:
        pairs = [(0, 1)]

    changes_list = []
    for u, v in pairs:
        changes = int(cn_profile[u, 0] != cn_profile[v, 0])
        changes += np.sum(np.diff(cn_profile[u]) != np.diff(cn_profile[v])).item()
        changes_list.append(changes)
    return changes_list
