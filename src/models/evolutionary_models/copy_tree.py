import logging

import networkx as nx
import dendropy as dpy
import numpy as np
import scipy.stats as sp_stats

from models.evolutionary_models import EvoModel, h_eps, h_eps0


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
