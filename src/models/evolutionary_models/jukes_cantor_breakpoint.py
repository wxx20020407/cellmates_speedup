import itertools

import numpy as np
import scipy.special as sp
import scipy.stats as ss
import dendropy as dpy

from models.evolutionary_models import EvoModel, p_delta_trans_mat, p_delta_start_prob, get_zipping_mask0, \
    get_zipping_mask
from models.observation_models import ObsModel
from models.observation_models.read_counts_models import PoissonModel
from simulation.datagen import simulate_cn
from utils.tree_utils import label_tree


class JCBModel(EvoModel):

    def __init__(self, n_states, alpha: float = 1., **kwargs):
        self.alpha = alpha
        self._lengths = None
        self._trans_mat = None
        self._start_prob = None
        self.loglikelihood = None
        super().__init__(n_states=n_states)

    @property
    def lengths(self):
        return self._lengths

    @lengths.setter
    def lengths(self, value):
        self._lengths = value
        self._compute_transitions()

    @property
    def theta(self):
        return self.lengths

    @theta.setter
    def theta(self, value):
        self.lengths = value

    @property
    def trans_mat(self):
        """
        array shape (n_states,) * 6
        """
        return self._trans_mat

    @property
    def start_prob(self):
        """
        array shape (n_states,) * 3
        """
        return self._start_prob

    def update(self, exp_changes, exp_no_changes, **kwargs):
        zero_tol = kwargs.get('zero_tol', 1e-10)
        d, dp = exp_changes, exp_no_changes
        # update l according to formula
        # if l -> +inf, pDeltaDelta == pDeltaDelta'
        log_arg = 1 - self.n_states / (self.n_states - 1) * d / (dp + d)
        assert np.all(log_arg > 0)
        # if np.any(log_arg <= 0):
        #     logger.error(f"too many changes detected: D = {d}, D' = {dp}\n"
        #                       f"...saturating l for cells {v},{w}")
        l_i = - 1 / (self.alpha * self.n_states) * np.log(np.clip(log_arg, a_min=zero_tol, a_max=None))
        self.lengths = l_i

    def new(self):
        return JCBModel(self.n_states, self.alpha)

    def expected_changes(self, obs_vw, obs_model: ObsModel = None) -> tuple:
        """
        Compute the expected number of changes in the Jukes-Cantor model under
         over the copy number conditional distribution for the three branches:
         (r, u), (u, v), (u, w).

        $$
        D(l) = E_{p(C|Y,l)}[ \sum_{m=1}^M \sum_{i,i',j,j'} 1(i'-i \neq j'-j) ]
        $$
        """
        if obs_model is None:
            obs_model = PoissonModel(self.n_states, 100., 100.)
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
                    # eps_uw (sum over m, j, j'
                    pair_tsm = sp.logsumexp(log_xi, axis=(0, 2, 5))

                d[e] = np.exp(sp.logsumexp(pair_tsm[~comut_mask]))
                dp[e] = np.exp(sp.logsumexp(pair_tsm[comut_mask]))

        return d, dp, loglik


    def two_slice_marginals(self, obs_vw, obs_model: ObsModel = None) -> np.ndarray:
        """
        Computes the two slice marginals of a hidden markov model with three latent chains.
        Specifically, for each point m of the chain (site), and each pair of triplet states
        (i,j,k)[m] -> (i'j'k')[m+1], it computes the
            $$ \log P(X_m = (i,j,k), X_{m+1} = (i',j',k') | Y, \theta) $$
        Also computes the log likelihood of the observations in the forward pass (saved in self.loglikelihood attribute).
        Parameters
        ----------
        obs_vw array of shape (n_sites, 2)
        theta array of shape (3,) with the triplet parameters
        n_states number of copy number states
        jcb if True, use Jukes-Cantor-Breakpoint model, otherwise use the CopyTree model
        alpha float, alpha parameter for the JCB model, length scaling factor
        Notation as in wiki page: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm#Update

        Returns
        -------
        array of shape (n_sites - 1,) + (n_states,) * 6, log two slice marginals
        (n_sites -1, n_states, n_states, n_states, n_states, n_states, n_states)

        """
        n_states = self.n_states
        if obs_model is None:
            obs_model = PoissonModel(n_states, 100., 100.)
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

            #  -- ALTERATIVE IMPLEMENTATION --
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

    def _forward_pass_likelihood(self, obs_vw, log_emissions):
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

        assert np.allclose(sp.logsumexp(alpha, axis=(1, 2, 3)), np.zeros(n_sites))
        return alpha, log_likelihood

    def _compute_transitions(self):
        """
        Compute the transition matrix for the JCB model using the current lengths.
        """
        self._trans_mat = self._compute_trans_mat()
        self._start_prob = self._compute_start_prob()

    def _compute_trans_mat(self):
        n_states = self.n_states
        l_trip = self.lengths
        alpha = self.alpha

        trans_mat = np.empty((n_states,) * 6)
        # transition from r -> u (fix r = 2)
        a_ru = p_delta_trans_mat(n_states, l_trip[0], alpha=alpha)[:, :, 2, 2]
        a_uv = p_delta_trans_mat(n_states, l_trip[1], alpha=alpha)
        a_uw = p_delta_trans_mat(n_states, l_trip[2], alpha=alpha)
        # results is state (m-1) i, j, k -> (m) x, y, z
        trans_mat[...] = np.einsum('xi,yjxi,zkxi->ijkxyz', a_ru, a_uv, a_uw)
        return trans_mat

    def _compute_start_prob(self):
        n_states = self.n_states
        l_trip = self.lengths
        alpha = self.alpha

        trip_start = np.empty((n_states,) * 3)

        # transition from r -> u (fix r = 2)
        a_ru = p_delta_start_prob(n_states, l_trip[0], alpha=alpha)[:, 2]
        # transition from u -> v, w
        a_uv = p_delta_start_prob(n_states, l_trip[1], alpha=alpha)
        a_uw = p_delta_start_prob(n_states, l_trip[2], alpha=alpha)
        # results is state i, j, k
        trip_start[...] = np.einsum('i, ij, ik -> ijk', a_ru, a_uv, a_uw)
        return trip_start

    def simulate_data(self, n_sites: int, l_mean: float = None):
        """
        Simulate copy number profiles for a quadruplet.
        Parameters
        ----------
        n_sites, number of sites
        l_mean, mean edge length for the JCB model

        Returns
        -------
        np.ndarray with shape (4, n_sites) with copy number profiles
        """
        tree = dpy.Tree.get(data="((0,1)2)3;", schema='newick', taxon_namespace=dpy.TaxonNamespace(['0', '1']))
        label_tree(tree)
        tree.is_rooted = True
        # generate edge _lengths
        l_true = np.empty(3)
        if l_mean is None:
            l_true[:] = np.array([0.01, 0.03, 0.008])
        else:
            for i in range(3):
                l_true[i] = ss.expon(scale=l_mean / self.alpha).rvs()
        r, u, v, w = tuple(map(str, [3, 2, 0, 1]))
        for edge in tree.preorder_edge_iter():
            # centroid to root
            if edge.head_node.label == u:
                edge.length = l_true[0]
            # centroid to v
            elif edge.head_node.label == v:
                edge.length = l_true[1]
            # centroid to w
            elif edge.head_node.label == w:
                edge.length = l_true[2]

        # and cn profiles
        cn = simulate_cn(tree, n_sites, self.n_states, alpha=self.alpha)

        return cn




