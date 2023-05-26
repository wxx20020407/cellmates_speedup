import itertools
from typing import Optional, Union

import numpy as np
import scipy.stats as stats
from hmmlearn import hmm

from models.quadruplet import Quadruplet

from src.models.copy_tree import h_eps, get_zipping_mask, get_zipping_mask0


class EM():
    """
    Runs the EM-algorithm for a quadruplet. Requires the copy number sequence of the root and observations of a pair
    of leaves.
    """

    def __init__(self, quadruplet: Quadruplet):
        self.quadruplet = quadruplet

    def run_hmmlearn(self):
        yv = self.quadruplet.yv
        yw = self.quadruplet.yv
        A = self.quadruplet.A
        y = np.concatenate([yv, yw])
        lengths = [len(yv), len(yw)]
        eps_trans_matrix_prior = np.ones((A, A, A)) / A * 0.05

        lambdas_prior = np.ones((A, 2))
        init_prior = np.zeros((3, A))
        init_prior[:, 2] = 1.
        model = hmm.PoissonHMM(n_components=(A, A, A),
                               startprob_prior=init_prior,
                               transmat_prior=eps_trans_matrix_prior,
                               lambdas_prior=lambdas_prior,
                               lambdas_weight=0.0,
                               algorithm='viterbi',
                               random_state=None,
                               n_iter=10,
                               tol=0.01,
                               verbose=False,
                               params='stl', init_params='', implementation='log')
        model.startprob_ = init_prior
        model.fit(y, lengths)


def compute_log_emissions(obs_vw: np.ndarray, n_states) -> np.ndarray:
    """
    returns the log probability of the observations for each site, and each pair of copy number states (v, w)
    Parameters
    ----------
    obs_vw
    n_states

    Returns
    -------
    array of shape (n_sites, n_states, n_states), log p(y_m^{vw} | C_m^v = i, C_m^w = j) for each m, i, j
    """
    lam = 100.
    assert obs_vw.shape[1] == 2
    n_sites = obs_vw.shape[0]
    log_emissions = np.empty((n_sites, n_states, n_states))
    for m, i, j in itertools.product((range(n_sites), range(n_states), range(n_states))):
        # log p(y_m^v | . ) + log p(y_m^w | . )
        log_emissions[m, i, j] = stats.poisson.logpmf(obs_vw[m], lam * np.array([i, j])).sum()

    return log_emissions


def compute_trans_mat(eps_trip, n_states) -> np.ndarray:
    trans_mat = np.empty((n_states,) * 6)
    # transition from r -> u (fix r = 2)
    a_ru = h_eps(n_states, eps=eps_trip[0])[:, :, 2, 2]
    a_uv = h_eps(n_states, eps=eps_trip[1])
    a_uw = h_eps(n_states, eps=eps_trip[2])
    # results is state i, j, k -> x, y, z
    trans_mat = np.einsum('xi,yjxi,zkxi->ijkxyz', a_ru, a_uv, a_uw)
    return trans_mat


def compute_start_prob(eps_trip, n_states):
    return np.ones((n_states,) * 3) / n_states ** 3


def forward_pass(obs_vw, start_prob, trans_mat, log_emissions):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    alpha = np.empty((n_sites, n_states, n_states, n_states))
    # TODO: implement
    return alpha


def backward_pass(obs_vw, start_prob, trans_mat, log_emissions):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    beta = np.empty((n_sites, n_states, n_states, n_states))
    # TODO: implement
    return beta


def two_slice_marginals(obs_vw, epsilon_k, n_states):
    n_sites = obs_vw.shape[0]
    # define start_probs
    start_prob = compute_start_prob(epsilon_k, n_states)
    # define transitions
    trans_mat = compute_trans_mat(epsilon_k, n_states)

    # define emission prob (for emission pair)
    # log emission shape (n_sites, n_states, n_states)
    log_emissions = compute_log_emissions(obs_vw, n_states)
    # compute forward: shape (n_sites, n_states, n_states, n_states)
    alpha = forward_pass(obs_vw, start_prob, trans_mat, log_emissions)
    # compute backward: shape (n_sites, n_states, n_states, n_states)
    beta = backward_pass(obs_vw, start_prob, trans_mat, log_emissions)
    # tsm shape: (n_sites - 1,) + (n_states,) * 6
    # compute two slice: xi
    tsm = np.einsum('mijk,ijkxyz,yz,mxyz -> mijkxyz', alpha[:n_sites-1], trans_mat,
                    np.exp(log_emissions), beta[1:, ...])
    tsm /= np.einsum('mijk,mijk->', alpha, beta)

    return tsm


def update_eps(epsilon_k, tsm):
    raise NotImplementedError("update eps is still not implemented")

    n_states = tsm.shape[1]
    eps_new = np.empty(3)
    comut_mask0 = get_zipping_mask0(n_states)
    comut_mask = get_zipping_mask(n_states)
    for e in range(3):
        deps = 0.
        deps_prime = 1.
        pair_tsm = np.empty((n_states,) * 4)
        if e == 0:
            # eps_ru
            pair_tsm = tsm.sum(axis=(0, 2, 3, 5, 6))
            # use comut_masks
            # deps = pair_tsm[]
            # deps_prime = pair_tsm[]


        eps_new[e] = deps / (deps + deps_prime)

    return eps_new


def em_alg(obs: np.ndarray) -> np.ndarray:
    """
Implementation of algorithm 6 in write-up
    Parameters
    ----------
    obs array of shape (n_sites, n_cells)
    Returns
    -------
    array of shape (n_cells, n_cells), centroid-to-root distance
    """
    # params
    n_states = 7
    eps_init = (1, 10)
    n_sites, n_cells = obs.shape
    epsilon_hat = np.empty((n_cells, n_cells))

    # for each pair of cells
    for v, w in itertools.product(range(n_cells), repeat=2):
        # initialize eps = (eps_ru, eps_uv, eps_uw)
        epsilon_k = np.random.beta(1, 10, size=3)
        # triplet state u, v, w
        convergence = False
        while not convergence:
            # compute two slice marginals
            tsm = two_slice_marginals(obs[:, [v, w]], epsilon_k, n_states)  # shape (n_states, n_states, n_states)
            # update epsilon_k
            epsilon_kp1 = update_eps(epsilon_k, tsm)
            # check for convergence
            convergence = np.allclose(epsilon_k, epsilon_kp1)
            # update current eps
            epsilon_k = epsilon_kp1

        epsilon_hat[v, w] = epsilon_k[0]  # ctr distance is eps_ru (first of triplet)

    return epsilon_hat