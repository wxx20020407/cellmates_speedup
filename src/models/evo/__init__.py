import itertools
import logging
import random

import dendropy as dpy
import networkx as nx
import numpy as np
from scipy import special as sp, stats as sp_stats, stats as ss

from models.evo.basefunc import get_zipping_mask, get_zipping_mask0, p_delta_change, p_delta_trans_mat, \
    p_delta_start_prob, h_eps, h_eps0

from models.obs import ObsModel, PoissonModel
from utils.tree_utils import label_tree


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
        array shape (n_states,) * 6
        """
        return self._trans_mat


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
        assert np.all(log_arg > 0)
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
        cn = self.simulate_cn(tree, n_sites)

        return cn
