import itertools
import logging

import networkx as nx
import numpy as np
import scipy.stats as sp_stats
from scipy.special import logsumexp

from models.evolutionary_models import EvoModel, h_eps, h_eps0
from models.observation_models.read_counts_models import PoissonModel


class CopyTree(EvoModel):
    def __init__(self, n_states, true_tree: nx.DiGraph = None):
        if true_tree is None:
            true_tree = nx.DiGraph([(0, 1), (1, 2), (1, 3)])
        self.true_tree = true_tree
        self.K = len(true_tree.nodes)
        self._eps = None
        self._trans_mat = None
        self._start_prob = None
        super().__init__(n_states=n_states)

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        self._eps = value
        self._compute_transitions()

    @property
    def theta(self):
        return self.eps

    @theta.setter
    def theta(self, value):
        self.eps = value

    @property
    def trans_mat(self):
        return self._trans_mat

    @property
    def start_prob(self):
        return self._start_prob

    def simulate_data(self, eps_a, eps_b, eps_0, M: int = 100):
        eps = np.zeros((self.K, self.K))
        logging.debug(f'Copy Tree data simulation - eps_a: {eps_a}, eps_b: {eps_b}, eps_0:{eps_0} ')
        eps_dist = sp_stats.beta(eps_a, eps_b)
        if eps_dist.std()**2 > 0.1 * eps_dist.mean():
            logging.warning(
                f'Large variance for epsilon: {eps_dist.std()**2} (mean: {eps_dist.mean()}. Consider increasing '
                f'eps_b param.')

        tree = self.true_tree
        for u, v in tree.edges:
            eps[u, v] = eps_dist.rvs()
            tree.edges[u, v]['weight'] = eps[u, v]

        # generate copy numbers
        c = np.empty((self.K, M), dtype=int)
        c[0, :] = 2 * np.ones(M, )
        h_eps0_cached = h_eps0(self.n_states, eps_0)
        for u, v in nx.bfs_edges(tree, source=0):
            t0 = h_eps0_cached[c[u, 0], :]
            c[v, 0] = np.argmax(sp_stats.multinomial(n=1, p=t0).rvs())
            h_eps_uv = h_eps(self.n_states, eps[u, v])
            for m in range(1, M):
                # j', j, i', i
                transition = h_eps_uv[:, c[v, m - 1], c[u, m], c[u, m - 1]]
                c[v, m] = np.argmax(sp_stats.multinomial(n=1, p=transition).rvs())

        return eps, c

    def new(self):
        return CopyTree(self.n_states, self.true_tree)

    def update(self, exp_changes, exp_no_changes, **kwargs):
        raise NotImplementedError

    def expected_changes(self, **kwargs) -> tuple:
        """
        Returns
        -------
        expected changes and no changes and log likelihood (d, d', loglik)
        """
        raise NotImplementedError

    def _compute_transitions(self):
        self._trans_mat = self._compute_trans_mat()
        self._start_prob = self._compute_start_prob()

    def _compute_trans_mat(self):
        n_states = self.n_states
        eps_trip = self.eps

        trans_mat = np.empty((n_states,) * 6)
        # transition from r -> u (fix r = 2)
        a_ru = h_eps(n_states, eps=eps_trip[0])[:, :, 2, 2]
        a_uv = h_eps(n_states, eps=eps_trip[1])
        a_uw = h_eps(n_states, eps=eps_trip[2])
        # results is state i, j, k -> x, y, z
        trans_mat[...] = np.einsum('xi,yjxi,zkxi->ijkxyz', a_ru, a_uv, a_uw)
        return trans_mat

    def _compute_start_prob(self):
        # TODO: uniform prob, but not implemented properly yet
        return np.ones((self.n_states,) * 3) / self.n_states ** 3


# TODO: insert the following code into the CopyTree class
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


def em_alg(obs: np.ndarray, n_states: int = 7, lam=100) -> np.ndarray:
    """
Implementation of algorithm 6 in write-up
    Parameters
    ----------
    obs array of shape (n_sites, n_cells)
    n_states int, number of copy number states
    Returns
    -------
    array of shape (n_cells, n_cells), centroid-to-root distance
    """
    # params
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
            log_xi, loglik = two_slice_marginals(obs[:, [v, w]], epsilon_k, n_states, lam=lam)
            # update epsilon_k
            epsilon_kp1 = update_eps(log_xi)
            # check for convergence
            convergence = np.allclose(epsilon_k, epsilon_kp1, rtol=1e-2)
            # update current eps
            epsilon_k[...] = epsilon_kp1

        epsilon_hat[v, w] = epsilon_k[0]  # ctr distance is eps_ru (first of triplet)

    return epsilon_hat

def two_slice_marginals(obs_vw, theta: np.ndarray, n_states: int, jcb: bool = False, alpha: float = 1., lam=100) \
        -> tuple[np.ndarray, float]:
    """
    Computes the two slice marginals of a hidden markov model with three latent chains.
    Specifically, for each point m of the chain (site), and each pair of triplet states
    (i,j,k)[m] -> (i'j'k')[m+1], it computes the
        $$ \log P(X_m = (i,j,k), X_{m+1} = (i',j',k') | Y, \theta) $$
    Also outputs the log likelihood of the observations which is computed in the forward pass.
    Parameters
    ----------
    obs_vw array of shape (n_sites, 2)
    theta array of shape (3,) with the triplet parameters
    n_states number of copy number states
    jcb if True, use Jukes-Cantor-Breakpoint model, otherwise use the CopyTree model
    alpha float, alpha parameter for the JCB model, length scaling factor

    Returns
    -------
    tuple of array of shape (n_sites - 1,) + (n_states,) * 6, log two slice marginals and float, log likelihood

    """
    n_sites = obs_vw.shape[0]
    # define start_probs and transitions depending on chosen model
    start_prob, trans_mat = get_start_transition_probs(alpha, jcb, n_states, theta)

    # define emission prob (for emission pair)
    # log emission shape (n_sites, n_states, n_states)
    log_emissions = compute_log_emissions(obs_vw, n_states, lam=lam)
    # compute forward: shape (n_sites, n_states, n_states, n_states)
    alpha_probs, loglik = _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions)
    # compute backward: shape (n_sites, n_states, n_states, n_states)
    beta_probs = backward_pass(obs_vw, trans_mat, log_emissions)
    # tsm shape: (n_sites - 1,) + (n_states,) * 6
    # compute two slice: xi
    log_xi = alpha_probs[(np.arange(n_sites - 1), ...) + (np.newaxis,) * 3] + \
             np.log(trans_mat)[np.newaxis, ...] + \
             log_emissions[(np.arange(1, log_emissions.shape[0]),) + (np.newaxis,) * 4 + (...,)] + \
             beta_probs[(np.arange(1, beta_probs.shape[0]), ...) + (np.newaxis,) * 3]
    log_xi -= np.expand_dims(logsumexp(log_xi, axis=tuple(range(1, 7))), axis=tuple(range(1, 7)))

    assert np.allclose(logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))

    return log_xi, loglik

def compute_log_emissions(obs_vw: np.ndarray, n_states, lam: float = 100, pois_mean_eps=1e-5) -> np.ndarray:
    """
    returns the log probability of the observations for each site, and each pair of copy number states (v, w)
    Parameters
    ----------
    obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
    n_states, number of copy number states
    lam, per-copy mean read count
    pois_mean_eps, small constant to avoid log(0)

    Returns
    -------
    array of shape (n_sites, n_states, n_states), log p(y_m^{vw} | C_m^v = i, C_m^w = j) for each m, i, j
    """
    assert obs_vw.shape[1] == 2
    model = PoissonModel(n_states, lambda_v_prior=lam, lambda_w_prior=lam)
    return model.log_emission(obs_vw)



def get_start_transition_probs(alpha, jcb, n_states, theta):
    start_prob = compute_eps_start_prob(theta, n_states)
    trans_mat = compute_eps_trans_mat(theta, n_states)
    return start_prob, trans_mat


def compute_eps_trans_mat(eps_trip, n_states) -> np.ndarray:
    evo_model = CopyTree(n_states=n_states)
    evo_model.eps = eps_trip
    return evo_model.trans_mat


def compute_eps_start_prob(eps_trip, n_states):
    evo_model = CopyTree(n_states=n_states)
    evo_model.eps = eps_trip
    return evo_model.start_prob


def _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions, eps=1e-5) -> (np.ndarray, float):
    """
    Compute the forward pass of the hidden markov model with three latent chains and return the forward probabilities
    as well as the log likelihood of the observations. See wrapper function `forward_pass` for more details.
    """
    # transmat idx = (i, j, k, x, y, z) = (m-1, m)
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    alpha = np.empty((n_sites, n_states, n_states, n_states))

    # init alpha[0] = start_prob * emission[0] (in log-form)
    alpha[0, ...] = np.log(start_prob.clip(eps)) + log_emissions[0, np.newaxis, ...]
    alpha[0] -= logsumexp(alpha[0])

    log_likelihood = 0.
    for m in range(1, n_sites):
        # [ \sum_{x,y,z} alpha_{m-1}(x, y, z) p(i, j, k | x, y, z) ] p(y_m^{vw} | i, j, k)
        log_acc = -np.inf * np.ones((n_states, n_states, n_states))
        for x, y, z in itertools.product(range(n_states), repeat=3):
            log_acc = np.logaddexp(alpha[m - 1, x, y, z] + np.log(trans_mat[x, y, z, ...]), log_acc)
        alpha[m, ...] = log_acc + log_emissions[m, np.newaxis, ...]
        # normalization step
        norm = logsumexp(alpha[m])
        alpha[m] -= norm
        # update log likelihood to save computation
        log_likelihood += norm

    assert np.allclose(logsumexp(alpha, axis=(1, 2, 3)), np.zeros(n_sites))
    return alpha, log_likelihood


def forward_pass(obs_vw, start_prob, trans_mat, log_emissions, eps=1e-5):
    """
    Compute the forward pass of the hidden markov model with three latent chains and return the forward probabilities.

    Parameters
    ----------
    obs_vw array of shape (n_sites, 2) with observations for pair of leaves
    start_prob array of shape (n_states, n_states, n_states) with start probabilities
    trans_mat array of shape (n_states, n_states, n_states, n_states, n_states, n_states) with transition probabilities
    log_emissions array of shape (n_sites, n_states, n_states) with log emissions
    eps float, small constant to avoid log(0)

    Returns
    -------
    array of shape (n_sites, n_states, n_states, n_states), forward probabilities

    """
    return _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions, eps)[0]


def backward_pass(obs_vw, trans_mat, log_emissions):
    n_sites = obs_vw.shape[0]
    n_states = trans_mat.shape[0]
    beta = np.empty((n_sites, n_states, n_states, n_states))
    # initialize beta[M] = 1. (in log form) - normalized
    beta[-1, ...] = - 3 * np.log(n_states)

    # compute iteratively
    for m in reversed(range(n_sites - 1)):
        # -- ORIGINAL IMPLEMENTATION --
        # beta[m, ...] = logsumexp(beta[m + 1, ...] +
        #                          np.log(trans_mat)
        #                          + log_emissions[m + 1, np.newaxis, np.newaxis, np.newaxis, ...],
        #                          axis=(3, 4, 5))

        #  -- ALTERATIVE IMPLEMENTATION --
        log_acc = -np.inf * np.ones((n_states, n_states, n_states))
        for x, y, z in itertools.product(range(n_states), repeat=3):
            log_acc = np.logaddexp(beta[m + 1, x, y, z] +
                                   np.log(trans_mat[..., x, y, z]) +
                                   log_emissions[m + 1, y, z],  # ]
                                   log_acc)
        beta[m, ...] = log_acc

        # normalization step
        beta[m] -= logsumexp(beta[m])

    return beta


def likelihood(obs_vw, theta, n_states, lam=100, alpha=1., jcb=True) -> float:
    start_prob, trans_mat = get_start_transition_probs(alpha, jcb, n_states, theta)
    log_emissions = compute_log_emissions(obs_vw, n_states, lam=lam)
    return _forward_pass_likelihood(obs_vw, start_prob, trans_mat, log_emissions)[1]
