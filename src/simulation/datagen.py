"""
Synthetic data generation functions.
"""
import logging
from itertools import combinations
from typing import TypedDict

import numpy as np
import scipy.stats as ss
import dendropy as dpy
import random
import anndata

from models.evolutionary_models import p_delta_change, EvoModel
from models.evolutionary_models.copy_tree import CopyTree
from models.evolutionary_models.jukes_cantor_breakpoint import JCBModel
from models.observation_models import ObsModel
from models.observation_models.normalized_read_counts_models import NormalModel
from models.observation_models.read_counts_models import PoissonModel
from utils.math_utils import l_from_p, p_from_l
from utils.tree_utils import random_binary_tree, get_node2node_distance, label_tree


class Dataset(TypedDict):
    """
    Data structure for synthetic data.
    """
    obs: np.ndarray
    tree: dpy.Tree
    cn: np.ndarray

def simulate_quadruplet(n_sites,
                        obs_model: ObsModel | str = 'poisson',
                        evo_model: EvoModel | str = 'jcb',
                        gamma_params: tuple | list[tuple] = (1, 1),
                        n_states: int = None) -> Dataset:
    """
    Simulate a quadruplet tree with 2 leaves, one internal node and a root.
    The tree is rooted and the edge_lengths are generated from an exponential distribution if l_mean is None,
    otherwise they are set to a fixed value: [0.01, 0.03, 0.008].
    The copy number profiles are simulated from the tree and the observations are emitted from the leaves.
    Indices are r, u, v, w = 3, 2, 0, 1.
    Parameters
    ----------
    n_sites: int, number of sites
    alpha: float, alpha parameter for evolution model
    l_mean: float, mean of the exponential distribution for edge _lengths
    gamma_params: tuple or list of tuples, parameters for gamma distribution for edge _lengths
        if tuple, the same parameters are used for all edges, otherwise a list of tuples is expected, one for each edge
    n_states: int, number of copy number states (if None, it will use the evolution model parameters)

    Returns
    -------
    dict with keys 'obs', 'tree', 'cn'

    """
    # generate tree
    # generate dendropy tree with 2 leaves, one internal node and root
    tree = dpy.Tree.get(data="((0,1)2)3;", schema='newick', taxon_namespace=dpy.TaxonNamespace(['0', '1']))
    label_tree(tree)
    tree.is_rooted = True
    evo_model = EvoModel.get_instance(evo_model, n_states)
    obs_model = ObsModel.get_instance(obs_model, n_states)

    if isinstance(gamma_params, tuple):
        gamma_params = [gamma_params] * 3
    if isinstance(gamma_params, list):
        if len(gamma_params) == 1:
            gamma_params = gamma_params * 3
        elif len(gamma_params) == 2:
            gamma_params.append(gamma_params[1])
        elif len(gamma_params) > 3:
            logging.error(f"too many gamma_params provided, using the first 3")
            gamma_params = gamma_params[:3]
    assert len(gamma_params) == 3 and all(len(gamma_params[i]) == 2 for i in range(3)), "gamma_params must be a tuple or list of 3 tuples"
    gamma_params = np.stack(gamma_params)

    # simulate edge_lengths (or epsilon param for 'copytree')
    edge_lengths = ss.gamma.rvs(gamma_params[:, 0], scale=gamma_params[:, 1])
    if isinstance(evo_model, CopyTree):
        edge_lengths = p_from_l(edge_lengths, n_states=n_states)
    for edge in tree.preorder_edge_iter():
        # centroid to root
        if edge.head_node.label == '2':
            edge.length = edge_lengths[0]
        # centroid to v
        elif edge.head_node.label == '0':
            edge.length = edge_lengths[1]
        # centroid to w
        elif edge.head_node.label == '1':
            edge.length = edge_lengths[2]

    # simulate copy number profiles
    cn = evo_model.simulate_cn(tree, n_sites)

    # emit observations from tree leaves
    obs = obs_model.sample(cn[2:4, :])

    return {
        'obs': obs,
        'tree': tree,
        'cn': cn
    }

def emit_normalized_obs(cn_seq, mu=1.0, scale=1.0):
    eps = ss.norm(loc=0., scale=scale).rvs(size=len(cn_seq))
    return np.clip(cn_seq / mu + eps, a_min=0., a_max=None)


def emit_raw_obs(cn_seq, lam=100.):
    return ss.poisson.rvs(mu=np.clip(cn_seq, a_min=.01, a_max=None) * lam, size=len(cn_seq))


def get_root_distance(centroid):
    root_distance = 0
    while centroid.parent_node is not None:
        root_distance += centroid.edge_length
        centroid = centroid.parent_node
    return root_distance


def get_ctr_table(tree: dpy.Tree) -> np.ndarray:
    n_cells = len(tree.leaf_nodes())
    assert n_cells == len(tree.taxon_namespace)
    ctr_table = - np.ones((n_cells, n_cells, 3))
    for r, s in combinations(range(n_cells), 2):
        assert r < s, "r must be less than s to ensure upper triangular matrix"
        # most recent common ancestor
        centroid = tree.mrca(taxon_labels=[str(r), str(s)])
        ctr_table[r, s, 0] = get_root_distance(centroid)
        ctr_table[r, s, 1] = get_node2node_distance(tree, centroid.label, str(r))
        ctr_table[r, s, 2] = get_node2node_distance(tree, centroid.label, str(s))

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


def rand_dataset(n_cells: int, n_states: int, n_sites: int, evo_model: EvoModel | str = 'jcb',
                 obs_model: ObsModel | str = 'normal', alpha=1.,
                 p_change: float = .2, seed=None) -> Dataset:
    # generate random sc binary tree
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        dpy.utility.GLOBAL_RNG.seed(seed)

    # parse evo_model and obs_model params
    evo_model = EvoModel.get_instance(evo_model, n_states)
    obs_model = ObsModel.get_instance(obs_model, n_states)

    # seed already set, no need to set it again
    tree = random_binary_tree(n_cells, length_mean=l_from_p(p_change, n_states) / alpha, seed=None)
    # set tree to rooted
    tree.is_rooted = True
    # simulate copy number chains
    cn = evo_model.simulate_cn(tree, n_sites)
    # emit observations from tree leaves
    obs = np.empty((n_sites, n_cells), dtype=np.float64)
    for t in tree.leaf_node_iter():
        cell_id = int(t.label)
        obs[:, cell_id] = obs_model.sample(cn[cell_id])

    # return dict with observations and all latent variables
    data = {
        'obs': obs,
        'tree': tree,  # contains _lengths as edges
        'cn': cn
    }
    return data
