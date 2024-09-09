"""
Synthetic data generation functions.
"""
import logging
from itertools import combinations
from typing import TypedDict

import numpy as np
import scipy.stats as ss
import dendropy
import random
import anndata

from models.copy_tree import p_delta_change
from utils.math_utils import l_from_p
from utils.tree_utils import random_binary_tree, get_node2node_distance


class Dataset(TypedDict):
    """
    Data structure for synthetic data.
    """
    obs: np.ndarray
    tree: dendropy.Tree
    cn: np.ndarray


def simulate_cn_seq(prev_cn, n_states, l, alpha=1.):
    node_cn = np.empty_like(prev_cn)
    # scale l if needed
    pdd = p_delta_change(n_states, l, change=False, alpha=alpha)
    # simulate first copy number
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


def emit_normalized_obs(cn_seq, mu=1.0, scale=1.0):
    eps = ss.norm(loc=0., scale=scale).rvs(size=len(cn_seq))
    return np.clip(cn_seq / mu + eps, a_min=0., a_max=None)


def emit_raw_obs(cn_seq, lam=100.):
    return ss.poisson.rvs(mu=np.clip(cn_seq, a_min=.01, a_max=None) * lam, size=len(cn_seq))


def simulate_cn(tree, n_sites, n_states, alpha=1.):
    cn = np.empty((len(tree.nodes()), n_sites))
    cn[tree.seed_node.label, :] = 2
    # tree needs index-labeled node
    assert tree.seed_node.label is not None
    for n in tree.preorder_node_iter():
        if n != tree.seed_node:
            cn[n.label] = simulate_cn_seq(cn[n.parent_node.label], n_states, n.edge_length, alpha=alpha)
    return cn


def label_tree(tree):
    """
    Assigns labels to tree nodes. Leaves are assigned with cell ids, internal nodes with decremental numbers
    different from cell ids.
    """
    rev_node_idx = len(tree.nodes()) - 1
    for n in tree.nodes():
        if n.is_leaf():
            n.label = int(n.taxon.label)
        else:
            n.label = rev_node_idx
            rev_node_idx -= 1


def get_root_distance(centroid):
    root_distance = 0
    while centroid.parent_node is not None:
        root_distance += centroid.edge_length
        centroid = centroid.parent_node
    return root_distance


def get_ctr_table(data: Dataset) -> np.ndarray:
    n_cells = data['obs'].shape[1]
    ctr_table = np.zeros((n_cells, n_cells, 3))
    for r, s in combinations(range(n_cells), 2):
        assert r < s, "r must be less than s to ensure upper triangular matrix"
        # most recent common ancestor
        centroid = data['tree'].mrca(taxon_labels=[str(r), str(s)])
        ctr_table[r, s, 0] = get_root_distance(centroid)
        ctr_table[r, s, 1] = get_node2node_distance(data['tree'], centroid.label, r)
        ctr_table[r, s, 2] = get_node2node_distance(data['tree'], centroid.label, s)

    return ctr_table


def rand_ann_dataset(n_cells: int, n_states: int, n_sites: int, **kwargs):
    #   using different hmms for each chromosome
    # kwargs = [alpha, obs_type]
    # minimum n_sites = 200 because bins are shared among 23 chromosomes
    # in the human genome
    if n_sites < 200:
        logging.debug(f"requested bins number {n_sites} is too low for human genome, setting n_sites = 200")
        n_sites = 200
    anndataset = anndata.AnnData()
    # TODO: implement
    #   using different hmms for each chromosome
    data = rand_dataset(n_cells, n_states, n_sites, **kwargs)

    return anndataset


def rand_dataset(n_cells: int, n_states: int, n_sites: int, alpha=1., obs_type='norm', p_change: float = .2,
                 seed=None) -> Dataset:
    # generate random sc binary tree
    tree = random_binary_tree(n_cells, length_mean=l_from_p(p_change, n_states), seed=seed)
    label_tree(tree)
    # set tree to rooted
    tree.is_rooted = True
    # simulate copy number chains
    cn = simulate_cn(tree, n_sites, n_states, alpha=alpha)
    # emit observations from tree leaves
    obs = np.empty((n_sites, n_cells))
    for t in tree.leaf_node_iter():
        cell_id = t.label
        if obs_type == 'pois':
            obs[:, cell_id] = emit_raw_obs(cn[cell_id])
        elif obs_type == 'norm':
            obs[:, cell_id] = emit_normalized_obs(cn[cell_id], scale=.7)
        else:
            logging.debug(f"type {obs_type} not supported for obs model")

    # return dict with observations and all latent variables
    data = {
        'obs': obs,
        'tree': tree,  # contains lengths as edges
        'cn': cn
    }
    return data
