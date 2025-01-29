import itertools
import logging
import math
import random

import dendropy as dpy
import numpy as np
import scipy.special as sp
from scipy import special as sp

from models.observation_models import ObsModel
from models.observation_models.read_counts_models import PoissonModel


class EvoModel:

    def __init__(self, n_states, **kwargs):
        self._start_prob = None
        self._trans_mat = None
        self.loglikelihood = None
        self.n_states = n_states

    @property
    def theta(self):
        return False

    @theta.setter
    def theta(self, value):
        pass

    def simulate_data(self, **kwargs):
        pass

    def simulate_cn(self, tree: dpy.Tree, n_sites: int) -> np.ndarray:
        pass

    def sample_cn_child(self, prev_cn, l: float = None, alpha=1., p_change: float = None):
        """
        Simulate a copy number sequence from a previous sequence.
        Parameters
        ----------
        prev_cn: np.ndarray, previous copy number sequence
        l: float, edge length parameter
        p_change: float, probability of change
        alpha: float, alpha parameter for evolution model

        Returns
        -------
        np.ndarray, shape (n_sites,) simulated copy number sequence
        """
        # validate params
        assert l is not None or p_change is not None, "either l or p_change must be provided"
        assert l is None or p_change is None, "only one of l or p_change must be provided"

        node_cn = np.empty_like(prev_cn)
        # scale l if needed
        if p_change is None:
            pdd = p_delta_change(self.n_states, l, change=False, alpha=alpha)
        else:
            pdd = 1 - p_change

        # simulate first copy number
        u = random.random()
        if u < pdd:
            node_cn[0] = prev_cn[0]
        else:
            node_cn[0] = random.choice([j for j in range(self.n_states) if j != prev_cn[0]])

        for m in range(1, len(prev_cn)):
            u = random.random()
            no_change_cn = prev_cn[m] - prev_cn[m - 1] + node_cn[m - 1]
            if prev_cn[m] == 0:
                # 0 absorption
                no_change_cn = 0

            if 0 <= no_change_cn < self.n_states:
                if u < pdd:
                    node_cn[m] = no_change_cn
                else:
                    node_cn[m] = random.choice([j for j in range(self.n_states) if j != no_change_cn])
            elif no_change_cn < 0:
                node_cn[m] = 0
            else:
                node_cn[m] = random.choice([j for j in range(self.n_states)])
        return node_cn

    def expected_changes(self, obs_vw, obs_model: ObsModel = None) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the expected number of changes which over the copy number conditional distribution for the three branches:
         (r, u), (u, v), (u, w).
         This function depends on the specific instance of the model only
         through the two slice marginals, which will is overridden in the subclass.

        $$
        D(l) = E_{p(C|Y,l)}[ \sum_{m=1}^M \sum_{i,i',j,j'} 1(i'-i \neq j'-j) ]
        $$
        where $p(C|Y,l)$ is determined by the specific model from which this function is called.
        """
        if obs_model is None:
            obs_model = PoissonModel(self.n_states, 100., 100.)
        else:
            assert obs_model.n_states == self.n_states

        d = np.empty(3)
        dp = np.empty_like(d)
        # compute two slice marginals
        # prob(Cm = ijk, Cm+1 = i'j'k' | Y)
        log_xi = self.two_slice_marginals(obs_vw, obs_model=obs_model)
        loglik = self.loglikelihood  # computed in the two slice marginals (forward pass)

        comut_mask0 = get_zipping_mask0(self.n_states).transpose()
        comut_mask = get_zipping_mask(self.n_states).transpose(tuple(reversed(range(4))))
        # count expected changes/no-changes
        for e in range(3):
            if e == 0:
                # l_ru (sum over m, j, k, j', k')
                pair_tsm = sp.logsumexp(log_xi, axis=(0, 2, 3, 5, 6))
                # use comut_mask0
                d[0] = np.exp(sp.logsumexp(pair_tsm[~comut_mask0]))
                dp[0] = np.exp(sp.logsumexp(pair_tsm[comut_mask0]))
            else:
                if e == 1:
                    # eps_uv (sum over m, i, i')
                    pair_tsm = sp.logsumexp(log_xi, axis=(0, 3, 6))
                else:
                    # eps_uw (sum over m, j, j')
                    pair_tsm = sp.logsumexp(log_xi, axis=(0, 2, 5))

                d[e] = np.exp(sp.logsumexp(pair_tsm[~comut_mask]))
                dp[e] = np.exp(sp.logsumexp(pair_tsm[comut_mask]))

        return d, dp, loglik

    def new(self):
        pass

    def update(self, exp_changes, exp_no_changes, **kwargs):
        pass

    @classmethod
    def get_instance(cls, evo_model, n_states):
        # avoid circular import
        from models.evolutionary_models.copy_tree import CopyTree
        from models.evolutionary_models.jukes_cantor_breakpoint import JCBModel

        if isinstance(evo_model, EvoModel):
            if evo_model.n_states != n_states:
                logging.warning(f"Number of states mismatch: {evo_model.n_states} != {n_states},"
                                f" keeping n_states = {evo_model.n_states} from the model object")
            return evo_model
        elif evo_model == 'copytree':
            return CopyTree(n_states=n_states)
        elif evo_model == 'jcb':
            return JCBModel(n_states=n_states)
        else:
            raise ValueError(f"Unknown evolutionary model {evo_model}")

    def two_slice_marginals(self, obs_vw, obs_model: ObsModel) -> np.ndarray:
        """
        Computes the two slice marginals of a hidden markov model with three latent chains.
        Specifically, for each point m of the chain (site), and each pair of triplet states
        (i,j,k)[m] -> (i'j'k')[m+1], it computes the
            $$ \log P(X_m = (i,j,k), X_{m+1} = (i',j',k') | Y, \theta) $$
        Also computes the log likelihood of the observations in the forward pass (saved in self.loglikelihood attribute).
        Parameters
        ----------
        obs_vw array of shape (n_sites, 2)
        obs_model instance of ObsModel

        Returns
        -------
        array of shape (n_sites - 1,) + (n_states,) * 6, log two slice marginals
        (n_sites -1, n_states, n_states, n_states, n_states, n_states, n_states)

        """
        n_states = self.n_states
        assert obs_model.n_states == n_states
        n_sites = obs_vw.shape[0]

        # define emission prob (for emission pair)
        # log emission shape (n_sites, n_states, n_states)
        log_emission = obs_model.log_emission(obs_vw)
        # compute forward: shape (n_sites, n_states, n_states, n_states)
        alpha_probs, self.loglikelihood = self._forward_pass_likelihood(obs_vw, log_emission)
        # compute backward: shape (n_sites, n_states, n_states, n_states)
        beta_probs = self.backward_pass(obs_vw, log_emission)
        # tsm shape: (n_sites - 1,) + (n_states,) * 6
        # compute two slice: xi
        log_xi = alpha_probs[(np.arange(n_sites - 1), ...) + (np.newaxis,) * 3] + \
                 np.log(self.trans_mat)[np.newaxis, ...] + \
                 log_emission[(np.arange(1, log_emission.shape[0]),) + (np.newaxis,) * 4 + (...,)] + \
                 beta_probs[(np.arange(1, beta_probs.shape[0]), ...) + (np.newaxis,) * 3]
        log_xi -= np.expand_dims(sp.logsumexp(log_xi, axis=tuple(range(1, 7))), axis=tuple(range(1, 7)))

        assert np.allclose(sp.logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))
        return log_xi

    def backward_pass(self, obs_vw, log_emissions):
        n_sites = obs_vw.shape[0]
        n_states = self.trans_mat.shape[0]
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

            #  -- ALTERNATIVE IMPLEMENTATION --
            log_acc = -np.inf * np.ones((n_states, n_states, n_states))
            for x, y, z in itertools.product(range(n_states), repeat=3):
                log_acc = np.logaddexp(beta[m + 1, x, y, z] +
                                       np.log(self.trans_mat[..., x, y, z]) +
                                       log_emissions[m + 1, y, z],  # ]
                                       log_acc)
            beta[m, ...] = log_acc

            # normalization step
            beta[m] -= sp.logsumexp(beta[m])

        return beta

    def _forward_pass_likelihood(self, obs_vw, log_emissions) -> tuple[np.ndarray, float]:
        """
        Compute the forward pass of the hidden markov model with three latent chains and return the forward probabilities
        as well as the log likelihood of the observations.

        Parameters
        ----------
        obs_vw array of shape (n_sites, 2) with observations for pair of leaves
        log_emissions array of shape (n_sites, n_states, n_states) with log emissions

        Returns
        -------
        tuple with alpha array of shape (n_sites, n_states, n_states, n_states) with forward probabilities and
        log likelihood of the observations
        """
        eps = 1e-10  # to avoid log(0)
        # transmat idx = (i, j, k, x, y, z) = (m-1, m)
        n_sites = obs_vw.shape[0]
        n_states = self.trans_mat.shape[0]
        alpha = np.empty((n_sites, n_states, n_states, n_states))

        # init alpha[0] = start_prob * emission[0] (in log-form)
        alpha[0, ...] = np.log(self.start_prob.clip(eps)) + log_emissions[0, np.newaxis, ...]
        alpha[0] -= sp.logsumexp(alpha[0])

        log_likelihood = 0.
        for m in range(1, n_sites):
            # [ \sum_{x,y,z} alpha_{m-1}(x, y, z) p(i, j, k | x, y, z) ] p(y_m^{vw} | i, j, k)
            log_acc = -np.inf * np.ones((n_states, n_states, n_states))
            for x, y, z in itertools.product(range(n_states), repeat=3):
                log_acc = np.logaddexp(alpha[m - 1, x, y, z] + np.log(self.trans_mat[x, y, z, ...]), log_acc)
            alpha[m, ...] = log_acc + log_emissions[m, np.newaxis, ...]
            # normalization step
            norm = sp.logsumexp(alpha[m])
            alpha[m] -= norm
            # update log likelihood to save computation
            log_likelihood += norm

        assert (np.allclose(sp.logsumexp(alpha, axis=(1, 2, 3)), np.zeros(n_sites)),
                f"Forward pass normalization error:{sp.logsumexp(alpha, axis=(1, 2, 3))}")
        return alpha, log_likelihood

    @property
    def start_prob(self):
        """
        array shape (n_states,) * 3
        """
        return self._start_prob

    @property
    def trans_mat(self):
        """
        array shape (n_states,) * 6
        """
        return self._trans_mat


def get_zipping_mask(n_states) -> np.ndarray:
    """Build a mask on the i - i' == j - j' condition

    Parameters
    ----------
    n_states :
        Number of total copy number states (cn = 0, 1, ..., n_states - 1)
    Returns
    -------
    np boolean tensor of shape (n_states, n_states, n_states, n_states),
    true where the indices satisfy the condition.
    Idx order is `mask[j', j, i', i]`
    """
    ind_arr = np.indices((n_states, n_states))
    # i - j
    imj = ind_arr[0] - ind_arr[1]
    # i - j == k - l
    mask = imj == imj[:, :, np.newaxis, np.newaxis]
    return mask


def get_zipping_mask0(n_states) -> np.ndarray:
    """Build a mask on the i == j condition

    Parameters
    ----------
    n_states :
        Number of total copy number states (cn = 0, 1, ..., n_states - 1)
    Returns
    -------
    np boolean tensor of shape (n_states, n_states),
    true where the indices satisfy the condition
    """
    ind_arr = np.indices((n_states, n_states))
    # i = j (diagonal)
    mask = ind_arr[0] == ind_arr[1]
    return mask


def p_delta(n_states, l, i, ii, j, jj, alpha=1.):
    """
    Returns the transition probability given the length and the
    copy number configurations.
    Parameters
    ----------
    n_states int, number of copy number states 0, ..., K (equals K + 1) with K max cn
    l float, length parameter
    i int, C_{m-1}^p
    ii int, C_m^p
    j int, C_{m-1}^v
    jj int, C_m^v

    Returns
    -------
    normalized transition probability
    """
    if ii - i + j < 0 or ii - i + j > n_states - 1:
        return 1 / n_states
    else:
        change = ii - i != jj - j
        return p_delta_change(n_states, l, change, alpha=alpha)


def p_delta_change(n_states, l, change: bool, alpha: float = 1.):
    if not change:
        p_out = 1 / n_states + (n_states - 1) / n_states * math.exp(- n_states * alpha * l)
        # p_out = 1 / n_states + (n_states - 1) / n_states * math.exp(- n_states / (n_states - 1) * alpha * l)
    else:
        p_out = 1 / n_states - math.exp(- n_states * alpha * l) / n_states
        # p_out = 1 / n_states - math.exp(- n_states / (n_states - 1) * alpha * l) / n_states
    return p_out


def p_delta_trans_mat(n_states, l, alpha: float = 1.):
    """
    Indexing order: [j', j, i', i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        l: arc distance parameter

    Returns:
        tensor of shape (A x A x A x A) with A = n_states
    """
    mat = np.empty((n_states,) * 4)

    for (i, ii, j, jj) in itertools.product(range(n_states), repeat=4):
        mat[jj, j, ii, i] = p_delta(n_states, l, i, ii, j, jj, alpha=alpha)
    return mat


def p_delta_start_prob(n_states, l, alpha: float = 1.):
    """
    p_delta(C_1^v | C_1^p) initial probability
    Indexing order: [j, i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        l: arc distance parameter

    Returns:
        tensor of shape (A x A) with A = n_states
    """
    mat = np.empty((n_states,) * 2)

    for (i, j) in itertools.product(range(n_states), repeat=2):
        mat[j, i] = p_delta_change(n_states, l, j != i, alpha=alpha)
    return mat


def h_eps(n_states: int, eps: float) -> np.ndarray:
    """
Zipping function tensor for given epsilon. In arc u->v, for each
combination, P(Cv_m=j'| Cv_{m-1}=j, Cu_m=i', Cu_{m-1}=i) = h(j'|j, i', i).
Indexing order: [j', j, i', i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        eps: arc distance parameter

    Returns:
        tensor of shape (A x A x A x A) with A = n_states
    """
    # TODO: add zero-absorption P(Cu>0 | Cp=0) = 0
    mask_arr = get_zipping_mask(n_states=n_states)
    # put 1-eps where j'-j = i'-i
    a = mask_arr * (1 - eps)
    # put either 1 or 1-eps in j'-j != i'-i  and divide by the cases
    b = (1 - np.sum(a, axis=0)) / np.sum(~mask_arr, axis=0)
    # combine the two arrays
    out_arr = b * (~mask_arr) + a
    return out_arr


def h_eps0(n_states: int, eps0: float) -> np.ndarray:
    """
Simple zipping function tensor. P(Cv_1=j| Cu_1=i) = h0(j|i), indexing order: [j, i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        eps: arc distance parameter

    Returns:
        tensor of shape (A x A) with A = n_states
    """
    heps0_arr = eps0 / (n_states - 1) * np.ones((n_states, n_states))
    diag_mask = get_zipping_mask0(n_states)
    heps0_arr[diag_mask] = 1 - eps0
    return heps0_arr
