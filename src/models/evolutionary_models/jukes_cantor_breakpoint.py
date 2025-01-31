import numpy as np
import scipy.stats as ss
import dendropy as dpy

from models.evolutionary_models import EvoModel, p_delta_trans_mat, p_delta_start_prob
from utils.tree_utils import label_tree


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


