import io
import itertools
import logging
import random
import time
import unittest
from unittest.mock import MagicMock

from networkx.classes import neighbors

from cellmates.inference import neighbor_joining
from cellmates.inference.em import EM
from cellmates.utils import tree_utils, testing, visual

import dendropy
from Bio import Phylo
import networkx as nx
import numpy as np
from dendropy.calculate import treecompare
from matplotlib import pyplot as plt

from cellmates.simulation.datagen import rand_dataset
from cellmates.models.evo import JCBModel, SimulationEvoModel


class CellmatesTestCase(unittest.TestCase):
    """
    Tests for running Cellmates EM inference and tree reconstruction on simulated data.
    """

    def setUp(self) -> None:
        random.seed(0)
        np.random.seed(seed=0)

    def test_cellmates_simple_tree(self):
        # Inference parameters
        max_iter = 20
        tol = 1e-4

        # Simulation parameters
        n_sites = 100
        n_cells = 7
        n_states = 7
        n_clonal_events_per_edge = 3
        n_focal_events_per_edge = 5
        clonal_CN_length = 10
        obs_model = 'normal'
        sim_evo_model = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           n_focal_events=n_focal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length)

        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model,
                            evo_model=sim_evo_model)
        x = data['obs']
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)

        out_dir = testing.create_output_test_folder(sub_folder_name=f"Cellmates_EM_M_{n_sites}_N_{n_cells}")
        fig, ax = plt.subplots()
        visual.plot_cn_profile(cnps, ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # --------- Run Cellmates EM inference ---------
        evo_model = JCBModel(n_states=n_states)
        em_alg = EM(n_states, evo_model=evo_model, obs_model=obs_model)
        em_alg.fit(x, max_iter=max_iter, tol=tol, num_processors=20)

        distances = em_alg.distances
        print(f"Distance matrix: \n {distances[0, ...]}")

        # Get the inferred tree
        tree_res_nx = neighbor_joining.build_tree(distances)

        nx.write_network_text(tree_nx)
        nx.write_network_text(tree_res_nx)







