"""
Synthetic data generation functions.
"""
import logging

import networkx as nx
import numpy as np
import scipy.stats as ss
import dendropy
import random
import anndata

from models.copy_tree import p_delta_change


def simulate_cn_seq(prev_cn, n_states, l, alpha=1.):
    node_cn = np.empty_like(prev_cn)
    # scale l if needed
    pdd = p_delta_change(n_states, alpha * l, change=False)
    # simulate first copy number
    u = random.random()
    if u < pdd:
        node_cn[0] = prev_cn[0]
    else:
        node_cn[0] = random.choice([j for j in range(n_states) if j != prev_cn[0]])

    for m in range(1, len(prev_cn)):
        u = random.random()
        no_change_cn = prev_cn[m] - prev_cn[m-1] + node_cn[m-1]
        if 0 <= no_change_cn < n_states:
            if u < pdd:
                node_cn[m] = no_change_cn
            else:
                node_cn[m] = random.choice([j for j in range(n_states) if j != no_change_cn])
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
    cn[0, :] = 2
    # tree needs index-labeled node
    assert tree.seed_node.label is not None
    for n in tree.preorder_node_iter():
        if n.label != 0:
            cn[n.label] = simulate_cn_seq(cn[n.parent_node.label], n_states, n.edge_length, alpha=alpha)
    return cn


def label_tree(tree):
    for i, n in enumerate(tree.preorder_node_iter()):
        n.label = i


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


def rand_dataset(n_cells: int, n_states: int, n_sites: int, alpha=0.02, obs_type='norm') -> dict:
    # generate random sc binary tree
    tree = dendropy.treesim.treesim.birth_death_tree(.9, .4, num_extant_tips=n_cells)
    # ref: https://dendropy.org/primer/treesims.html
    # generate lengths
    # (done in dendropy)
    label_tree(tree)
    # simulate copy number chains
    # TODO: find best alpha for n_sites and delete argument from function (hide from out the function)
    cn = simulate_cn(tree, n_sites, n_states, alpha=alpha)
    # emit observations from tree leaves
    obs = np.empty((n_sites, n_cells))
    tax_id_map = {}
    for i, t in enumerate(tree.leaf_node_iter()):
        tax_id_map[t.taxon] = i
        if obs_type == 'pois':
            obs[:, i] = emit_raw_obs(cn[t.label])
        elif obs_type == 'norm':
            obs[:, i] = emit_normalized_obs(cn[t.label], scale=.7)
        else:
            logging.debug(f"type {obs_type} not supported for obs model")

    # return dict with observations and all latent variables
    data = {
        'obs': obs,
        'tree': tree,  # contains lengths as edges
        'cn': cn,
        'tax_id_map': tax_id_map
    }
    return data