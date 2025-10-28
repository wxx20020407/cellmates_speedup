"""
Utils for HMM functions (forward, backward, viterbi, etc.)
"""
import numpy as np
import numba
import scipy as sp

from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal, Poisson
from utils.math_utils import l_from_p


# wrapper class, parametrized by triplet probs eps_ru, eps_rv, eps_rw
class TripHMM:
    def __init__(self, n_states, obs_model: str, eps: tuple = None, lengths: tuple | None = None):
        self.n_states = n_states
        self.obs_model = obs_model
        self.eps = eps
        self.lengths = lengths
        # validata input
        if eps is None and lengths is None:
            raise ValueError("Either eps or lengths must be provided")
        if eps is not None and lengths is not None:
            raise ValueError("Only one of eps or lengths must be provided")
        self.hmm = self._build_hmm()

    def _build_hmm(self):
        # convert lengths to eps if needed
        if self.eps is None and self.lengths is not None:
            self.eps = l_from_p(self.lengths, n_states=self.n_states)

    def forward_backward(self, X, alg='pomegranate'):
        log_emissions = self.compute_log_emissions(X)
        if alg == 'pomegranate':
            return _forward_backward_pomegranate(log_emissions=log_emissions)
        else:
            raise ValueError(f"Unknown algorithm: {alg}")

    def _forward_backward_pomegranate(self, X):
        pass

    def compute_log_emissions(self, X):
        # use the obs_model to compute log emissions
        # TODO: implement
        pass


# helpers for pomegranate HMM
def pmg_make_normal_obs_models(n_states, mu, tau):
    obs_models = []
    for ijk in range(n_states ** 3):
        i = ijk // (n_states ** 2)
        j = (ijk // n_states) % n_states
        k = ijk % n_states
        mean_j = mu * j
        mean_k = mu * k
        # observation is a pair y_j, y_k ~ N(mu*j, tau) , N(mu*k, tau) independently
        obs_models.append(Normal((mean_j, mean_k), (tau, tau), covariance_type='diag'))
    return obs_models


def pmg_make_normal_emissions(X, n_states, mu, tau):
    n_samples = 1
    seq_length = X.shape[0]
    emissions = np.zeros((n_samples, seq_length, n_states ** 3))
    # for each m and each possible state ijk, compute the log prob of observing X[m] given state ijk
    for ijk in range(n_states ** 3):
        i = ijk // (n_states ** 2)
        j = (ijk // n_states) % n_states
        k = ijk % n_states
        mean_j = mu * j
        mean_k = mu * k
        # observation is a pair y_j, y_k ~ N(mu*j, tau) , N(mu*k, tau) independently
        obs_model = Normal((mean_j, mean_k), (tau, tau), covariance_type='diag')
        emissions[0, :, ijk] = obs_model.log_probability(X)
    return emissions

def pmg_convert_emissions(log_emissions):
    """
    From shape (n_sites, n_states, n_states) to (1, n_sites, n_states**3) to match pomegranate HMM input
    0th dim is batch size (1)
    """
    n_sites, n_states, _ = log_emissions.shape
    emissions_2D = log_emissions.reshape((n_sites, n_states ** 3))
    emissions_3D = emissions_2D[None, :, :]
    return emissions_3D

def _forward_likelihood_pomegranate(log_emissions, trans_mat, start_prob):
    """
    Compute forward variables and log likelihood using pomegranate HMM
    Parameters
    ----------
    log_emissions : np.ndarray
        Log emissions of shape (n_sites, n_states, n_states) as of the TripHMM model
    trans_mat : np.ndarray
        Transition matrix of shape (n_states, n_states, n_states, n_states, n_states, n_states)
    start_prob : np.ndarray
        Start probabilities of shape (n_states, n_states, n_states)
    Returns
    -------
    alpha : np.ndarray
        Forward variables of shape (n_sites, n_states, n_states, n_states)
    log_p : float
        Log likelihood of the observed sequence
    """
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    start_prob_1D = start_prob.flatten()
    model = DenseHMM(edges=trans_mat_2D, starts=start_prob_1D, ends=np.ones(n_states ** 3) / (n_states ** 3))
    f = model.forward(emissions=log_emissions_3D).numpy()
    alpha = f.reshape((n_sites, n_states, n_states, n_states)).numpy()
    log_p = sp.logsumexp(f[:, -1], dim=1)
    return alpha, log_p

def _forward_backward_pomegranate(log_emissions, trans_mat, start_prob):
    """
    Compute expected counts, marginal probabilities, and log likelihood using pomegranate HMM
    Parameters
    ----------
    """
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    start_prob_1D = start_prob.flatten()
    model = DenseHMM(edges=trans_mat_2D, starts=start_prob_1D, ends=np.ones(n_states ** 3) / (n_states ** 3))
    expected_counts, marginal, _, _, log_p = model.forward_backward(emissions=log_emissions_3D)
    expected_counts = expected_counts.reshape((n_states,) * 6).numpy()  # from 2D* (1, n_states**3, n_states**3) to 6D (n_states,) * 6
    marginal = marginal.reshape((n_sites, n_states, n_states, n_states)).numpy()  # from 2D* (1, n_sites, n_states**3) to 4D (n_sites, n_states, n_states, n_states)
    return expected_counts, marginal, log_p.item()

def _forward_likelihood_broadcast(log_emissions, trans_mat, start_prob):
    eps = 1e-10  # to avoid log(0)
    # transmat idx = (i, j, k, x, y, z) = (m-1, m)
    n_sites, n_states, _ = log_emissions.shape
    alpha = np.empty((n_sites, n_states, n_states, n_states))

    # init alpha[0] = start_prob * emission[0] (in log-form)
    alpha[0, ...] = np.log(start_prob.clip(eps)) + log_emissions[0, None, ...]
    log_trans_mat = np.log(trans_mat)
    norm = sp.logsumexp(alpha[0])
    alpha[0] -= norm

    log_likelihood = norm
    for m in range(1, n_sites):
        log_acc = sp.logsumexp(alpha[m - 1, :, :, :, None, None, None] + log_trans_mat, axis=(0, 1, 2))
        alpha[m, ...] = log_acc + log_emissions[m, None, :, :]
        norm = sp.logsumexp(alpha[m])
        # normalization step
        alpha[m] -= norm

        # update log likelihood to save computation
        log_likelihood += norm

    return alpha, log_likelihood

def _backward_pass_broadcast(log_emissions, trans_mat):
    n_sites, n_states, _ = log_emissions.shape

    log_trans_mat = np.log(trans_mat)
    # initialize beta[M] = 1. (in log form)
    beta = np.zeros((n_sites, n_states, n_states, n_states))
    beta[-1, ...] = - 3 * np.log(n_states)

    # compute iteratively
    for m in reversed(range(n_sites - 1)):
        beta[m, ...] = sp.logsumexp(beta[m + 1, None, None, None, :, :, :] +
                                    log_trans_mat +
                                    log_emissions[m + 1, None, None, None, None, :, :], axis=(3, 4, 5))
        # normalization step
        beta[m] -= sp.logsumexp(beta[m])
        # print(f"beta{m} sum: {sp.logsumexp(beta[m])}")
    return beta

def _backward_pass_pomegranate(log_emissions, trans_mat):
    n_sites, n_states, _ = log_emissions.shape
    log_emissions_3D = pmg_convert_emissions(log_emissions)
    trans_mat_2D = trans_mat.reshape((n_states ** 3, n_states ** 3))
    model = DenseHMM(edges=trans_mat_2D, starts=np.ones(n_states ** 3) / (n_states ** 3), ends=np.ones(n_states ** 3) / (n_states ** 3))
    b = model.backward(emissions=log_emissions_3D).numpy()
    beta = b.reshape((n_sites, n_states, n_states, n_states)).numpy()
    return beta

def _forward_backward_broadcast(log_emissions, trans_mat, start_prob, debug=False):
    # define emission prob (for emission pair)
    # log emission shape (n_sites, n_states, n_states)
    n_sites, n_states, _ = log_emissions.shape
    # compute forward: shape (n_sites, n_states, n_states, n_states)
    alpha, log_p = _forward_likelihood_broadcast(log_emissions, trans_mat, start_prob)
    # compute backward: shape (n_sites, n_states, n_states, n_states)
    beta = _backward_pass_broadcast(log_emissions, trans_mat)
    # tsm shape: (n_sites - 1,) + (n_states,) * 6
    log_trans_mat = np.log(trans_mat)
    # compute two slice: xi
    log_xi = alpha[(np.arange(n_sites - 1), ...) + (np.newaxis,) * 3] + \
             log_trans_mat[np.newaxis, ...] + \
             log_emissions[(np.arange(1, log_emissions.shape[0]),) + (np.newaxis,) * 4 + (...,)] + \
             beta[(np.arange(1, beta.shape[0]), ...) + (np.newaxis,) * 3]
    expected_counts = sp.logsumexp(log_xi, axis=0) - log_p
    # log_xi = log_xi - sp.logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6), keepdims=True) # check if faster
    log_gamma = sp.logsumexp(log_xi, axis=(4, 5, 6)) # Check axis for gamma
    # last site needs to be added separately
    log_alpha_final = alpha[-1, ...]
    log_evidence = sp.logsumexp(log_alpha_final)
    log_gamma_final = log_alpha_final - log_evidence
    log_gamma_final_expanded = log_gamma_final[np.newaxis, ...]
    log_gamma = np.concatenate([log_gamma, log_gamma_final_expanded], axis=0)
    if debug:
        assert np.allclose(sp.logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))
        assert np.allclose(sp.logsumexp(log_gamma, axis=(1, 2, 3)), np.zeros(n_sites))
    return expected_counts, log_gamma, log_p
