import logging
import random

import dendropy as dpy
import networkx as nx
import numpy as np
from scipy import special as sp, stats as sp_stats, stats as ss

from cellmates.models.evo.basefunc import get_zipping_mask, get_zipping_mask0, p_delta_change, p_delta_trans_mat, \
    p_delta_start_prob, h_eps, h_eps0

from cellmates.models.obs import ObsModel, PoissonModel
from cellmates.utils import tree_utils


class EvoModel:

    def __init__(self, n_states, **kwargs):
        self._start_prob = None
        self._trans_mat = None
        self.loglikelihood = None
        self.log_xi = None
        self.log_gamma = None
        self.n_states = n_states
        # optional parameters
        self.zero_absorption = kwargs.get('zero_absorption', False)
        self.focal_rate = kwargs.get('focal_rate', 0.)
        self.event_length_ratio = kwargs.get('event_length_ratio', 0.2)
        self.chromosome_ends = kwargs.get('chromosome_ends', []) # list of chromosome end positions (0-indexed) excluding last position
        self.debug = kwargs.get('debug', False)

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

        node_cn[:] = _evolve_cn_event_pois(prev_cn, pdd, self.n_states, zero_absorption=self.zero_absorption,
                                           focal_rate=self.focal_rate, event_length_ratio=self.event_length_ratio)
        # node_cn[:] = _evolve_cn_event_chain(prev_cn, pdd, self.n_states)  # old method
        return node_cn

    def _expected_changes(self, obs_vw, obs_model: ObsModel = None) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the expected number of changes which over the copy number conditional distribution for the three branches:
         (r, u), (u, v), (u, w).

        $$
        D(l) = E_{p(C|Y,l)}[ \sum_{m=1}^M \sum_{i,i',j,j'} 1(i'-i \neq j'-j) ]
        $$
        where $p(C|Y,l)$ is determined by the specific model from which this function is called.
        """
        if obs_model is None:
            obs_model = PoissonModel(self.n_states, 100., 100.)
        else:
            assert obs_model.n_states == self.n_states

        d = np.empty(3)  # expected changes
        dp = np.empty(3) # expected no changes
        # compute two slice marginals
        # prob(Cm = ijk, Cm+1 = i'j'k' | Y)
        log_xi, log_gamma = self.two_slice_marginals(obs_vw, obs_model=obs_model)
        loglik = self.loglikelihood  # computed in the two slice marginals (forward pass)
        self.log_xi = log_xi
        self.log_gamma = log_gamma
        log_gamma_1 = log_gamma[0]  # first site marginal prob C1 = ijk

        comut_mask0 = get_zipping_mask0(self.n_states).transpose()
        comut_mask = get_zipping_mask(self.n_states).transpose(tuple(reversed(range(4))))
        # count expected changes/no-changes
        for e in range(3):
            if e == 0:
                # l_ru (sum over m, j, k, j', k')
                pair_tsm = sp.logsumexp(log_xi, axis=(0, 2, 3, 5, 6))
                # use comut_mask0
                d[0] = np.exp(sp.logsumexp(pair_tsm[~comut_mask0]))  # change
                dp[0] = np.exp(sp.logsumexp(pair_tsm[comut_mask0]))  # no change
                # add first state
                pair_osm_1 = sp.logsumexp(log_gamma_1, axis=(1, 2))
                exp_p2 = np.exp(pair_osm_1[2])  # p(C1u = 2)
                d[0] += 1 - exp_p2  # change from r=2 to u != 2
                dp[0] += exp_p2 # no change from r=2 to u=2
            else:
                if e == 1:
                    # eps_uv (sum over m, i, i')
                    pair_tsm = sp.logsumexp(log_xi, axis=(0, 3, 6))
                    pair_osm_1 = sp.logsumexp(log_gamma_1, axis=2)
                else:
                    # eps_uw (sum over m, j, j')
                    pair_tsm = sp.logsumexp(log_xi, axis=(0, 2, 5))
                    pair_osm_1 = sp.logsumexp(log_gamma_1, axis=1)

                d[e] = np.exp(sp.logsumexp(pair_tsm[~comut_mask]))
                dp[e] = np.exp(sp.logsumexp(pair_tsm[comut_mask]))
                # add first state
                d[e] += np.exp(sp.logsumexp(pair_osm_1[~comut_mask0]))
                dp[e] += np.exp(sp.logsumexp(pair_osm_1[comut_mask0]))

        return d, dp, loglik

    def multi_chr_expected_changes(self, obs_vw, obs_model: ObsModel = None) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the expected number of changes which over the copy number conditional distribution for the three branches:
         (r, u), (u, v), (u, w). This function handles multiple chromosomes by splitting the observations
        at the chromosome ends and summing the expected changes over the chromosomes.
        If no chromosome ends are provided, it behaves like _expected_changes.
        Parameters
        ----------
        obs_vw: array of shape (n_sites, 2) with observations for pair of leaves
        obs_model: instance of ObsModel
        Returns
        -------
        tuple with expected changes, expected no changes and log likelihood
        """

        if obs_model is None:
            obs_model = PoissonModel(self.n_states, 100., 100.)
        else:
            assert obs_model.n_states == self.n_states
        # check chromosome ends against obs_vw shape
        if self.chromosome_ends:
            assert self.chromosome_ends[-1] < obs_vw.shape[0], "chromosome ends exceed number of observations"
        d = np.zeros(3)  # expected changes
        dp = np.zeros(3) # expected no changes
        loglik = 0.
        chr_start = 0
        for chr_end in self.chromosome_ends + [obs_vw.shape[0]]:
            chr_obs = obs_vw[chr_start:chr_end]
            chr_d, chr_dp, chr_loglik = self._expected_changes(chr_obs, obs_model=obs_model)
            d += chr_d
            dp += chr_dp
            loglik += chr_loglik
            chr_start = chr_end
        self.loglikelihood = loglik
        return d, dp, loglik

    def new(self):
        pass

    def update(self, exp_changes, exp_no_changes, **kwargs):
        pass

    # @classmethod
    # def get_instance(cls, evo_model, n_states):
    #     # avoid circular import
    #     from models.evo.copy_tree import CopyTree
    #     from models.evo.jukes_cantor_breakpoint import JCBModel
    #
    #     if isinstance(evo_model, EvoModel) and n_states is not None:
    #         if evo_model.n_states != n_states:
    #             logging.warning(f"Number of states mismatch: {evo_model.n_states} != {n_states},"
    #                             f" keeping n_states = {evo_model.n_states} from the model object")
    #         return evo_model
    #     elif evo_model == 'copytree':
    #         return CopyTree(n_states=n_states)
    #     elif evo_model == 'jcb':
    #         return JCBModel(n_states=n_states)
    #     else:
    #         raise ValueError(f"Unknown evolutionary model {evo_model}")

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
        # log_xi = log_xi - sp.logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6), keepdims=True) # check if faster
        log_gamma = sp.logsumexp(log_xi, axis=(4, 5, 6)) # Check axis for gamma
        # last site needs to be added separately
        log_alpha_final = alpha_probs[-1, ...]
        log_evidence = sp.logsumexp(log_alpha_final)
        log_gamma_final = log_alpha_final - log_evidence
        log_gamma_final_expanded = log_gamma_final[np.newaxis, ...]
        log_gamma = np.concatenate([log_gamma, log_gamma_final_expanded], axis=0)
        if self.debug:
            assert np.allclose(sp.logsumexp(log_xi, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1))
            assert np.allclose(sp.logsumexp(log_gamma, axis=(1, 2, 3)), np.zeros(n_sites))
        return log_xi, log_gamma

    def backward_pass(self, obs_vw, log_emissions, normalization=True) -> np.ndarray:
        n_sites = obs_vw.shape[0]
        n_states = self.trans_mat.shape[0]

        # initialize beta[M] = 1. (in log form)
        beta = np.zeros((n_sites, n_states, n_states, n_states))
        if normalization:
            beta[-1, ...] = - 3 * np.log(n_states)

        # compute iteratively
        for m in reversed(range(n_sites - 1)):
            beta[m, ...] = sp.logsumexp(beta[m + 1, None, None, None, :, :, :] +
                                  np.log(self.trans_mat) +
                                  log_emissions[m + 1, None, None, None, None, :, :], axis=(3, 4, 5))
            # normalization step
            if normalization:
                beta[m] -= sp.logsumexp(beta[m])
            # print(f"beta{m} sum: {sp.logsumexp(beta[m])}")

        return beta

    def _forward_pass_likelihood(self, obs_vw, log_emissions, normalization=True) -> tuple[np.ndarray, float]:
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
        alpha[0, ...] = np.log(self.start_prob.clip(eps)) + log_emissions[0, None, ...]
        norm = sp.logsumexp(alpha[0])
        if normalization:
            alpha[0] -= norm

        log_likelihood = norm
        for m in range(1, n_sites):
            log_acc = sp.logsumexp(alpha[m - 1, :, :, :, None, None, None] + np.log(self.trans_mat), axis=(0, 1, 2))
            alpha[m, ...] = log_acc + log_emissions[m, None, :, :]
            norm = sp.logsumexp(alpha[m])
            if normalization:
                # normalization step
                alpha[m] -= norm

            # update log likelihood to save computation
            log_likelihood += norm

        if normalization:
            assert np.allclose(sp.logsumexp(alpha, axis=(1, 2, 3)), np.zeros(n_sites)),\
                f"Forward pass normalization error:{sp.logsumexp(alpha, axis=(1, 2, 3))}"
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
        Indexing: p(x, y, z | i, j, k) = trans_mat[i, j, k, x, y, z]
        array shape (n_states,) * 6
        """
        return self._trans_mat

    def get_one_slice_marginals(self)-> tuple[np.ndarray, np.ndarray]:
        """
        Compute the one slice marginals from the one slice log gammas.
        """
        one_slice_marginal_v = np.einsum('mijk->mj', np.exp(self.log_gamma))
        one_slice_marginal_w = np.einsum('mijk->mk', np.exp(self.log_gamma))
        return one_slice_marginal_v, one_slice_marginal_w


class CopyTree(EvoModel):

    def __init__(self, n_states, **kwargs):
        self._eps = None
        super().__init__(n_states=n_states, **kwargs)

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        """
        Set the epsilon parameters for the CopyTree model and recompute the transition matrix.
        """
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

    def simulate_cn(self, tree: dpy.Tree, n_sites: int) -> np.ndarray:
        cn = np.empty((len(tree.nodes()), n_sites))
        cn[int(tree.seed_node.label), :] = 2
        # tree needs index-labeled node
        assert tree.seed_node.label is not None, "seed node must be labeled (rooted tree)"
        for n in tree.preorder_node_iter():
            if n != tree.seed_node:
                cn[int(n.label)] = self.sample_cn_child(cn[int(n.parent_node.label)], p_change=n.edge_length)
        return cn

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
        return CopyTree(self.n_states)

    def update(self, exp_changes, exp_no_changes, **kwargs) -> None:
        self.eps[:] = exp_changes / (exp_changes + exp_no_changes)

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
        # probability of cn u,v,w = i,j,k is the product of P(u = i), P(v=j | u=i) and P(w=k | u=i)
        # this can easily be done with einsum

        n_states = self.n_states
        trip_start = np.empty((n_states,) * 3)
        a_ru = h_eps0(n_states, self.eps[0])[:, 2]
        a_uv = h_eps0(n_states, self.eps[1])
        a_uw = h_eps0(n_states, self.eps[2])
        # results is state i, j, k
        trip_start[...] = np.einsum('i, ij, ik -> ijk', a_ru, a_uv, a_uw)
        return trip_start


class JCBModel(EvoModel):

    def __init__(self, n_states, alpha: float = 1., **kwargs):
        self.alpha = alpha
        self._lengths = None
        super().__init__(n_states=n_states, **kwargs)

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

    def update(self, exp_changes, exp_no_changes, **kwargs):
        zero_tol = kwargs.get('zero_tol', 1e-10)
        d, dp = exp_changes, exp_no_changes
        # update l according to formula
        # if l -> +inf, pDeltaDelta == pDeltaDelta'
        log_arg = 1 - self.n_states / (self.n_states - 1) * d / (dp + d)
        #assert np.all(log_arg > 0)
        # if np.any(log_arg <= 0):
        #     logger.error(f"too many changes detected: D = {d}, D' = {dp}\n"
        #                       f"...saturating l for cells {v},{w}")
        l_i = - 1 / (self.alpha * self.n_states) * np.log(np.clip(log_arg, a_min=zero_tol, a_max=None))
        self.lengths = l_i

    def new(self):
        return JCBModel(self.n_states, self.alpha)

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

    def simulate_cn(self, tree: dpy.Tree, n_sites: int) -> np.ndarray:
        cn = np.empty((len(tree.nodes()), n_sites))
        cn[int(tree.seed_node.label), :] = 2
        # tree needs index-labeled node
        assert tree.seed_node.label is not None, "seed node must be labeled (rooted tree)"
        for n in tree.preorder_node_iter():
            if n != tree.seed_node:
                cn[int(n.label)] = self.sample_cn_child(cn[int(n.parent_node.label)], n.edge_length, alpha=self.alpha)
        return cn

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
        tree_utils.label_tree(tree)
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
        cn = self.simulate_cn(tree, n_sites)

        return cn


def _evolve_cn_event_pois(prev_cn: np.ndarray, pdd: float, n_states: int, zero_absorption: bool = False,
                          focal_rate: float = 0., event_length_ratio: float = .2) -> np.ndarray:
    """
    Evolve copy number chain with focal and clonal events, modeling the length of the events with a Poisson distribution.
    The mean of the Poisson distribution is set to 5 for focal events and a fraction of the chain length M for clonal events.
    The number of events is sampled from a Poisson distribution with mean (1-pdd) * M / 2 so that the average number of breakpoints
    (start and end) is (1-pdd) * M.
    By default the focal event rate is 0 (focal events are disabled).
    Parameters
    ----------
    prev_cn: np.ndarray, previous copy number chain
    pdd: float, probability of no change
    n_states: int, number of copy number states
    zero_absorption: bool, if True, the zero absorption is enabled (no gains from 0)
    focal_rate: float, proportion of focal events over the total number of events
     (focal events are short, clonal events are long)
    event_length_ratio: float, ratio of the chain length to use as mean for the Poisson distribution
        of clonal event lengths. Default is 0.2 (20% of the chain length)
    Returns
    -------
    np.ndarray, shape (n_sites,) child copy number chain
    """
    M = prev_cn.shape[0]
    # focal events are short
    focal_length = 5 if M > 100 else max(2, M // 20)
    clonal_length_mean = max(10, int(M * event_length_ratio))
    # compute the number of events
    n_events = ss.poisson((1 - pdd) * M / 2).rvs()
    child_cn = prev_cn.copy()
    if n_events > 0:
        start_pos = np.random.permutation(M)[:n_events]
        start_pos.sort()
        for e in range(n_events):
            # sample 1 or -1
            delta = 1 if random.random() < 0.5 else -1
            start = int(start_pos[e])
            # focal event
            if focal_rate > 0 and random.random() < focal_rate:
                end = min(start + focal_length, M - 1)
            # clonal event
            else:
                length = ss.poisson(clonal_length_mean).rvs()
                end = min(start + length, M - 1)
            child_cn[start:end] = np.clip(child_cn[start:end] + delta, a_min=0, a_max=n_states - 1)
            if zero_absorption:
                zero_mask = prev_cn[start:end] == 0
                child_cn[start:end][zero_mask] = 0

    return child_cn

def _evolve_cn_event_chain(prev_cn: np.ndarray, pdd: float, n_states: int) -> np.ndarray:
    # simulate first copy number
    node_cn = np.empty_like(prev_cn)
    u = random.random()
    if u < pdd:
        node_cn[0] = prev_cn[0]
    else:
        node_cn[0] = random.choice([j for j in range(n_states) if j != prev_cn[0]])

    for m in range(1, len(prev_cn)):
        u = random.random()
        no_change_cn = prev_cn[m] - prev_cn[m - 1] + node_cn[m - 1]
        if prev_cn[m] == 0:
            # 0 absorption
            no_change_cn = 0

        if 0 <= no_change_cn < n_states:
            if u < pdd:
                node_cn[m] = no_change_cn
            else:
                node_cn[m] = random.choice([j for j in range(n_states) if j != no_change_cn])
        elif no_change_cn < 0:
            node_cn[m] = 0
        else:
            node_cn[m] = random.choice([j for j in range(n_states)])

    return node_cn

class SimulationEvoModel():
    """
    Model for simulating copy number evolution.
    Allows more control over simulated CN events, e.g., fixed number of events per edge, event lengths, etc.
    """

    def __init__(self,
                 clonal_CN_prob: float | dict = 0.05, clonal_CN_length_ratio: float = 0.2,
                 focal_prob: float | dict = 0.05, focal_length_avg: int = 5,
                 n_clonal_CN_events: int | dict = None, clonal_CN_length: int |dict = None,
                 n_focal_events: int | dict = None, focal_CN_length: int | dict = None,
                 allow_overlapping_CN_events=True, n_homoplasies=None, zero_absorption: bool = True):
        """
        Initialize simulation model.
        All dicts should have keys as edge tuples (u,v) where u is the parent node and v is the child node.
        Parameters
        ----------
        clonal_CN_prob: probability of clonal CN event per site per edge.
        clonal_CN_length_ratio: ratio of the CNP length used as mean for the clonal CN events.
        focal_prob: probability of focal CN event per site per edge.
        focal_length_avg: average length of focal CN events.
        n_clonal_CN_events: fixed number of clonal CN events per edge. If specified, disgards clonal_CN_prob.
        clonal_CN_length: fixed length of clonal CN events. If specified, disgards clonal_CN_length_ratio.
        n_focal_events: fixed number of focal CN events per edge. If specified, disgards focal_prob.
        focal_CN_length: fixed length of focal CN events. If specified, disgards focal_length_avg.
        allow_overlapping_CN_events: Only implemented for TRUE now.
        n_homoplasies: Not implemented yet.
        zero_absorption: Not implemented yet.
        """
        # ------- Simulation parameters -------
        # Clonal CN event parameters
        self.clonal_CN_prob = clonal_CN_prob
        self.clonal_CN_length_ratio = clonal_CN_length_ratio
        self.n_clonal_CN_events = n_clonal_CN_events
        self.clonal_CN_length = clonal_CN_length
        # Focal CN event parameters
        self.focal_prob = focal_prob
        self.focal_length_avg = focal_length_avg
        self.n_focal_events = n_focal_events
        self.focal_CN_length = focal_CN_length
        # Type of events parameters
        self.allow_overlapping_CN_events = allow_overlapping_CN_events # Only implemented for TRUE now
        self.n_homoplasies = n_homoplasies  # Not implemented yet
        self.zero_absorption = zero_absorption # Not implemented yet

        # Simulation helpers
        self.n_sites = None
        self.chr_idxs = None

        # Simulation outputs
        self.focal_events_out = {}
        self.clonal_events_out = {}
        self.clonal_CN_events_start_pos = {}
        self.focal_CN_events_start_pos = {}
        self.clonal_CN_events_end_pos = {}
        self.focal_CN_events_end_pos = {}

    def simulate_cn(self, tree: dpy.Tree, n_sites, chr_idxs=None)-> np.ndarray:
        self.n_sites = n_sites
        self.chr_idxs = chr_idxs
        nx_tree = tree_utils.convert_dendropy_to_networkx(tree)
        n_nodes = len(tree.nodes())
        root_idx = list(filter(lambda p: p[1] == 0, nx_tree.in_degree()))

        # initialize copy number array
        cn = np.empty((n_nodes, n_sites), dtype=int)
        cn.fill(2)  # root copy number is 2
        edges = list(tree.preorder_edge_iter())
        n_edges = len(edges)
        # Extend function to draw edges with homoplasies
        # Extend to WGD events

        for u,v in nx_tree.edges:
            n = tree.nodes()[v]
            if v == root_idx:
                continue
            else:
                n.cn = np.empty(n_sites, dtype=int)
                # Draw number of focal and clonal events
                n_clonal_events_uv, n_focal_events_uv = self.draw_number_of_CN_events(u, v)
                # Draw CN events (start, end) sites
                out_CN_pos = self.draw_CN_events_positions(u, v, n_clonal_events_uv, n_focal_events_uv, n_sites)
                clonal_start_pos, clonal_end_pos = out_CN_pos['clonal_start_pos'], out_CN_pos['clonal_end_pos']
                self.clonal_CN_events_start_pos[u,v] = clonal_start_pos
                self.clonal_CN_events_end_pos[u,v] = clonal_end_pos
                focal_start_pos, focal_end_pos = out_CN_pos['focal_start_pos'], out_CN_pos['focal_end_pos']
                self.focal_CN_events_start_pos[u,v] = focal_start_pos
                self.focal_CN_events_end_pos[u,v] = focal_end_pos

                # Inherit parent CNP
                n.cn[:] = cn[u, :]
                # Apply CN events to child CNP
                delta_CN_clonal_uv = self.draw_clonal_events(clonal_start_pos, clonal_end_pos)
                delta_CN_focal_uv = self.draw_focal_events(focal_start_pos, focal_end_pos)
                n.cn += delta_CN_clonal_uv + delta_CN_focal_uv
                n.cn = np.clip(n.cn, a_min=0, a_max=None)
                cn[v, :] = n.cn
        return cn

    def draw_number_of_CN_events(self, u, v):
        # Draw number of clonal CN events
        if self.n_clonal_CN_events is None:
            clonal_rate = self.clonal_CN_prob * self.n_sites
            n_clonal_events_uv = ss.poisson(clonal_rate)
        else:
            if isinstance(self.n_clonal_CN_events, dict):
                n_clonal_events_uv = self.n_clonal_CN_events[(u, v)]
            else:
                n_clonal_events_uv = self.n_clonal_CN_events
        # Draw number of focal CN events
        if self.n_focal_events is None:
            focal_rate = self.focal_prob * self.n_sites
            n_focal_events_uv = ss.poisson(focal_rate)
        else:
            if isinstance(self.n_focal_events, dict):
                n_focal_events_uv = self.n_focal_events[(u, v)]
            else:
                n_focal_events_uv = self.n_focal_events
        return n_clonal_events_uv, n_focal_events_uv

    def draw_CN_events_positions(self, u, v, n_clonal_events_uv, n_focal_events_uv, n_sites):
        if self.allow_overlapping_CN_events:
            clonal_start_pos = np.random.randint(0, n_sites, size=n_clonal_events_uv)
            focal_start_pos = np.random.randint(0, n_sites, size=n_focal_events_uv)
            # Draw clonal event lengths
            if self.clonal_CN_length is None:
                clonal_lengths = ss.poisson(self.clonal_CN_length_ratio * n_sites).rvs(size=n_clonal_events_uv)
            else:
                clonal_lengths = self.clonal_CN_length[u, v] if isinstance(self.clonal_CN_length, dict) else self.clonal_CN_length
            clonal_end_pos = np.clip(clonal_start_pos + clonal_lengths, a_min=None, a_max=n_sites - 1)

            # Draw focal event lengths
            if self.focal_CN_length is None:
                focal_lengths = ss.poisson(self.focal_length_avg).rvs(size=n_focal_events_uv)
            else:
                focal_lengths = self.focal_CN_length[u, v] if isinstance(self.focal_CN_length, dict) else self.focal_CN_length
            focal_end_pos = np.clip(focal_start_pos + focal_lengths, a_min=None, a_max=n_sites - 1)
            # Ensure no 0-absorption
            if self.zero_absorption:
                pass
        else:
            # Draw non-overlapping CN events
            raise NotImplementedError()

        out_dict = {
            'clonal_start_pos': clonal_start_pos,
            'clonal_end_pos': clonal_end_pos,
            'focal_start_pos': focal_start_pos,
            'focal_end_pos': focal_end_pos
        }
        return out_dict

    def draw_clonal_events(self, clonal_start_pos, clonal_end_pos):
        delta_CN_clonal_uv = np.zeros(self.n_sites, dtype=int)
        for s, e in zip(clonal_start_pos, clonal_end_pos):
            delta = 1 if random.random() < 0.5 else -1
            delta_CN_clonal_uv[s:e] += delta
        return delta_CN_clonal_uv

    def draw_focal_events(self, focal_start_pos, focal_end_pos):
        delta_CN_focal_uv = np.zeros(self.n_sites, dtype=int)
        for s, e in zip(focal_start_pos, focal_end_pos):
            delta = 1 if random.random() < 0.5 else -1
            delta_CN_focal_uv[s:e] += delta
        return delta_CN_focal_uv

