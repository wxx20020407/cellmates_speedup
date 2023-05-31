import itertools
from typing import Optional, Union

import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp
import networkx as nx
# from hmmlearn import hmm

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


def compute_log_emissions(obs_vw: np.ndarray, n_states, pois_mean_eps=1e-5) -> np.ndarray:
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
    for m, i, j in itertools.product(range(n_sites), range(n_states), range(n_states)):
        # log p(y_m^v | . ) + log p(y_m^w | . )
        log_emissions[m, i, j] = stats.poisson.logpmf(obs_vw[m],
                                                      np.clip(lam * np.array([i, j]), a_min=pois_mean_eps, a_max=None)).sum()

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
    # TODO: uniform prob, but not implemented properly yet
    return np.ones((n_states,) * 3) / n_states ** 3


def forward_pass(obs_vw, start_prob, trans_mat, log_emissions, eps=1e-5):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    alpha = np.empty((n_sites, n_states, n_states, n_states))

    # init alpha[0] = start_prob * emission[0] (in log-form)
    alpha[0, ...] = np.log(start_prob.clip(eps)) + log_emissions[0, np.newaxis, ...]
    alpha[0] -= logsumexp(alpha[0])

    for m in range(1, n_sites):
        alpha[m, ...] = logsumexp(alpha[m - 1] +
                                  np.log(trans_mat),
                                  axis=(0, 1, 2)) + log_emissions[m, np.newaxis, ...]
        # normalization step
        alpha[m] -= logsumexp(alpha[m])

    return alpha


def backward_pass(obs_vw, start_prob, trans_mat, log_emissions):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    beta = np.empty((n_sites, n_states, n_states, n_states))
    # initialize beta[M] = 1. (in log form) - normalized
    beta[-1, ...] = 3 * np.log(n_states)

    # compute iteratively
    for m in reversed(range(n_sites - 1)):
        beta[m, ...] = logsumexp(beta[m + 1, ...] +
                                 np.log(trans_mat)
                                 + log_emissions[m + 1, np.newaxis, np.newaxis, np.newaxis, ...],
                                 axis=(3, 4, 5))
        # normalization step
        beta[m] -= logsumexp(beta[m])

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
    log_xi = alpha[(np.arange(n_sites - 1), ...) + (np.newaxis,) * 3] + np.log(trans_mat)[np.newaxis, ...] + \
        log_emissions[(np.arange(1, log_emissions.shape[0]),) + (np.newaxis,) * 4 + (...,)] + \
        beta[(np.arange(1, beta.shape[0]), ...) + (np.newaxis,) * 3]
    log_xi -= np.expand_dims(logsumexp(log_xi, axis=tuple(range(1, 7))), axis=tuple(range(1, 7)))

    assert np.allclose(logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))

    return log_xi


def update_eps(log_xi):
    n_states = log_xi.shape[1]
    eps_new = np.empty(3)
    # get set `A` and `\neg A` as masks on the quadruplet
    # i - i' == j - j' (indexing: `mask[j', j, i', i]`)
    # after reverse: `mask[i, i', j, j']`
    comut_mask0 = get_zipping_mask0(n_states).transpose()
    comut_mask = get_zipping_mask(n_states).transpose(tuple(reversed(range(4))))
    for e in range(3):
        if e == 0:
            # eps_ru
            pair_tsm = logsumexp(log_xi, axis=(0, 2, 3, 5, 6))
            # use comut_mask0
            deps = np.exp(logsumexp(pair_tsm[~comut_mask0]))
            deps_prime = np.exp(logsumexp(pair_tsm[comut_mask0]))
        else:
            if e == 1:
                # eps_uv
                pair_tsm = logsumexp(log_xi, axis=(0, 3, 6))
            else:
                # eps_uw
                pair_tsm = logsumexp(log_xi, axis=(0, 2, 5))

            deps = np.exp(logsumexp(pair_tsm[~comut_mask]))
            deps_prime = np.exp(logsumexp(pair_tsm[comut_mask]))

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
    epsilon_hat = - np.ones((n_cells, n_cells))

    # for each pair of cells
    for v, w in itertools.combinations(range(n_cells), r=2):
        # initialize eps = (eps_ru, eps_uv, eps_uw)
        epsilon_k = np.random.beta(*eps_init, size=3)
        # triplet state u, v, w
        convergence = False
        while not convergence:
            # compute two slice marginals
            # shape (n_sites, n_states x 3, n_states x 3)
            log_xi = two_slice_marginals(obs[:, [v, w]], epsilon_k, n_states)
            # update epsilon_k
            epsilon_kp1 = update_eps(log_xi)
            # check for convergence
            convergence = np.allclose(epsilon_k, epsilon_kp1, rtol=1.e-2)
            # update current eps
            epsilon_k[...] = epsilon_kp1

        epsilon_hat[v, w] = epsilon_k[0]  # ctr distance is eps_ru (first of triplet)

    return epsilon_hat


def _build_tree_rec(ctr_table, cells: set, edges: set[tuple]):

    if len(cells) == 2:
        for c in cells:
            edges.add(('0', f'c{c}'))
    else:
        ctr_table_sub = ctr_table[np.ix_(list(cells), list(cells))]
        v, w = np.argwhere(ctr_table_sub == ctr_table_sub.max())[0].tolist()
        edges = _build_tree_rec(ctr_table, cells.difference({w}), edges)
        # find edge with cell v and add split
        for x, v_ in edges:
            if v_ == f'c{v}':
                edges = edges.difference((x, v_))
                u = x + '_0'
                edges = edges.union([(x, u), (u, v_), (u, f'c{w}')])
                break

    return edges


def build_tree(ctr_table):

    edges = _build_tree_rec(ctr_table, set(range(ctr_table.shape[0])), set())
    em_tree = nx.Graph()
    em_tree.add_edges_from(edges)

    return em_tree