"""
Synthetic data generation functions.
"""
import networkx as nx
import numpy as np
import scipy.stats as ss
import dendropy

def rand_dataset(n_cells: int, n_states: int, n_sites: int) -> dict:
    # generate random sc binary tree
    tree = dendropy.treesim.treesim.birth_death_tree(.8, .7, ntax=n_cells)
    # TODO: continue looking here https://dendropy.org/primer/treesims.html
    # generate lengths
    # simulate copy number chains
    # emit observations from tree leaves
    obs = np.empty((n_cells, n_sites))
    # return dict with observations and all latent variables
    data = {
        'obs': obs,
        'tree': tree,  # contains lengths as edges
    }
    return data