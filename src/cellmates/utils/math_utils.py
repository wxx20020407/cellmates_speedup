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


import numpy as np
from typing import Tuple, Dict, List


# --- Helpers (some are from the previous implementation) ---

def _get_log_probs(eps: float, K: int) -> Tuple[float, float]:
    """Helper to get log(1-eps) and log(eps/(K-1))"""
    C = K - 1
    if C == 0: return 0.0, -np.inf
    if eps == 0.0: return 0.0, -np.inf
    if eps == 1.0: return -np.inf, 0.0
    return np.log(1.0 - eps), np.log(eps / C)


def _flatten_state(l: int, p: int, o: int, K: int) -> int:
    """Converts a (l, p, o) state tuple to a single integer index."""
    return (l * K + p) * K + o


def _unflatten_state(flat_index: int, K: int) -> Tuple[int, int, int]:
    """Converts a flat state index back to a (l, p, o) tuple."""
    o = flat_index % K
    p = (flat_index // K) % K
    l = flat_index // (K * K)
    return l, p, o


def _build_log_A_t(
        K: int,
        Delta_r_t: int,
        log_probs_ru: Tuple[float, float],
        log_probs_uv: Tuple[float, float],
        log_probs_uw: Tuple[float, float]
) -> np.ndarray:
    """
    Vectorized O(K^6) function to build the full transition tensor.

    Returns:
        log_A_t (K, K, K, K, K, K): The tensor where
                                  log_A_t[k,m,n, l,p,o] is the transition prob.
    """
    # 1. Create 6D coordinate grids
    k, m, n, l, p, o = np.indices((K, K, K, K, K, K))

    # 2. Calculate deltas for all K^6 possibilities
    Delta_u = k - l
    Delta_v = m - p
    Delta_w = n - o

    # 3. Get log probs for each link
    log_p_ru_hi, log_p_ru_lo = log_probs_ru
    log_p_uv_hi, log_p_uv_lo = log_probs_uv
    log_p_uw_hi, log_p_uw_lo = log_probs_uw

    # 4. Build each log_P tensor using np.where
    log_P_ru = np.where(Delta_u == Delta_r_t, log_p_ru_hi, log_p_ru_lo)
    log_P_uv = np.where(Delta_v == Delta_u, log_p_uv_hi, log_p_uv_lo)
    log_P_uw = np.where(Delta_w == Delta_u, log_p_uw_hi, log_p_uw_lo)

    # 5. Sum them to get the final K^6 tensor
    log_A_t = log_P_ru + log_P_uv + log_P_uw

    return log_A_t


def viterbi_matrix_K6(
        log_emissions: np.ndarray,
        Z_r: np.ndarray,
        log_pi: np.ndarray,
        eps_ru: float,
        eps_uv: float,
        eps_uw: float
) -> Tuple[List[Tuple[int, int, int]], float]:
    """
    Performs the standard O(M*K^6) Viterbi algorithm using
    vectorized numpy operations instead of explicit for-loops.

    Returns:
        best_path: List of (l, p, o) tuples representing the most probable path
        max_log_prob: Log probability of the best path

    WARNING: This function is EXTREMELY memory-intensive.
    """
    M, K, _ = log_emissions.shape
    N = K ** 3  # Total number of composite states

    # --- 1. Get Log Probs for Transitions ---
    log_probs_ru = _get_log_probs(eps_ru, K)
    log_probs_uv = _get_log_probs(eps_uv, K)
    log_probs_uw = _get_log_probs(eps_uw, K)

    # --- 2. Initialization (t=0) ---
    log_delta = np.full((M, K, K, K), -np.inf)

    # psi stores the *flat index* of the best previous state
    psi = np.zeros((M, K, K, K), dtype=np.int32)

    log_emissions_t0 = log_emissions[0, None, :, :]  # Shape (1, K, K)
    log_delta[0] = log_pi + log_emissions_t0

    # --- 3. Recursion (t=1 to M-1) ---
    # This is the O(M * K^6) section

    for t in range(1, M):
        # --- This part replaces the 6-fold loop ---

        # 3a. Get the K^6 transition tensor
        Delta_r_t = Z_r[t - 1] - Z_r[t]
        log_A_t = _build_log_A_t(
            K, Delta_r_t, log_probs_ru, log_probs_uv, log_probs_uw
        )

        # 3b. Prepare log_delta[t-1] for broadcasting
        # Shape (k, m, n) -> (k, m, n, 1, 1, 1)
        log_delta_tm1_expanded = log_delta[t - 1][:, :, :, np.newaxis, np.newaxis, np.newaxis]

        # 3c. Compute the max-sum
        # This is log(delta_{t-1}(k,m,n)) + log(A_t(kmn, lpo))
        # Shape: (k, m, n, l, p, o)
        log_sum_tensor = log_delta_tm1_expanded + log_A_t

        # 3d. Flatten the source axes (k,m,n) to N = K^3
        # Shape: (K^3, K, K, K)
        flat_log_sum_tensor = log_sum_tensor.reshape((N, K, K, K))

        # 3e. Find the max log-prob and the argmax (backpointer)
        # We take the max over the source axis (axis=0)
        max_log_prob = np.max(flat_log_sum_tensor, axis=0)  # Shape (l, p, o)
        psi[t] = np.argmax(flat_log_sum_tensor, axis=0)  # Shape (l, p, o)

        # 3f. Store the final delta value
        # Shape (l, p, o) + (p, o) -> (l, p, o)
        log_delta[t] = max_log_prob + log_emissions[t, None, :, :]
        # --- End of loop replacement ---

    # --- 4. Termination ---
    max_log_prob = np.max(log_delta[M - 1])
    best_last_state_flat = np.argmax(log_delta[M - 1].reshape(N))

    # --- 5. Backtracking (same as before) ---
    best_path_flat = np.zeros(M, dtype=np.int32)
    best_path_flat[M - 1] = best_last_state_flat

    for t in range(M - 2, -1, -1):
        current_state_flat = best_path_flat[t + 1]
        l, p, o = _unflatten_state(current_state_flat, K)

        # Look up the flat index from the psi table
        prev_state_flat = psi[t + 1, l, p, o]
        best_path_flat[t] = prev_state_flat

    best_path = np.array([_unflatten_state(idx, K) for idx in best_path_flat])
    best_path_ordered = best_path[: [1, 2, 0]]  # reorder to (uv, uw, ru)
    return best_path_ordered, max_log_prob

