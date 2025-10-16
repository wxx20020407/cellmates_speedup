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
    eps = 1e-10 # to avoid log(0) and log(1) = -0.
    return - 1 / n_states * np.log(1 - n_states / (n_states - 1) * p_change + eps)

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

def get_expected_branch_lengths_from_cnps(cnps: np.ndarray, n_states: int, model='jcb'):
    """
    Estimate the expected branch lengths for CNP pairs from copy number profiles using the JCB or CopyTree model.
    Parameters
    ----------
    cnps: np.ndarray, shape (n_nodes, n_sites), copy number profiles for each node
    n_states: int, number of copy number states
    model: str, 'jcb' or 'copytree'

    Returns
    -------
    dict with keys as tuples of node pairs and values as expected branch lengths
    """
    n_nodes, n_sites = cnps.shape
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    changes_list = compute_cn_changes(cnps, pairs=pairs)

    expected_lengths = {}
    for (u, v), changes in zip(pairs, changes_list):
        if model == 'jcb':
            p_change = changes / n_sites
        elif model == 'copytree':
            p_change = changes / n_sites
        else:
            raise ValueError("Model must be 'jcb' or 'copytree'")
        expected_length = l_from_p(p_change, n_states=n_states)
        expected_lengths[(u, v)] = float(expected_length)

    return expected_lengths

