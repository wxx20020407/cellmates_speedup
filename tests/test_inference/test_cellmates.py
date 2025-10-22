import itertools
import logging
import os
import random
import unittest
from unittest.mock import MagicMock

import pytest
import anndata
import pandas as pd
import skbio
from Bio import Phylo
from dendropy.calculate import treecompare

from cellmates.common_helpers import cnasim_data
from cellmates.inference import neighbor_joining
from cellmates.inference.em import EM
from cellmates.utils import tree_utils, testing, visual, math_utils

import dendropy
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from cellmates.simulation.datagen import rand_dataset
from cellmates.models.evo import JCBModel, SimulationEvoModel


class CellmatesTestCase(unittest.TestCase):
    """
    Tests for running Cellmates EM inference and tree reconstruction on simulated data.
    """

    def setUp(self) -> None:
        self.seed = 0
        random.seed(self.seed)
        np.random.seed(seed=self.seed)
        dendropy.utility.GLOBAL_RNG.seed(self.seed)

    #@unittest.skip("This test only works when evo_model.new() is commented out in _fit_quadruplet in em.py")
    def test_cellmates_given_c(self):
        # Inference parameters
        max_iter = 20
        rtol = 1e-4

        # Simulation parameters
        n_sites = 1000
        n_cells = 20
        n_states = 7
        n_clonal_events_per_edge = 5
        n_focal_events_per_edge = 5
        clonal_CN_length = n_sites // 20
        obs_model_sim = 'normal'
        sim_evo_model = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           n_focal_events=n_focal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length)
        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model_sim,
                            evo_model=sim_evo_model)
        x = data['obs']
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)

        out_dir = testing.create_output_test_folder(sub_folder_name=f"M{n_sites}_N{n_cells}_A{n_states}")
        fig, ax = plt.subplots()
        visual.plot_cn_profile(cnps, cell_labels=np.arange(0, n_cells), ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # --------- Setup Models and Mock Expected changes D, D' ---------
        obs_model = obs_model_sim
        evo_model_temp = JCBModel(n_states=n_states)
        evo_model = JCBModel(n_states=n_states)
        cell_pairs = list(itertools.combinations(range(n_cells), r=2))
        D, Dp = testing.get_expected_changes(cnps, tree_nx, cell_pairs)
        exp_distances, exp_pairwise_distances = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

        fig, ax = plt.subplots()
        visual.plot_cell_pairwise_heatmap(exp_pairwise_distances, label=np.arange(0, n_cells), ax=ax)
        fig.savefig(out_dir + '/exp_pairwise_distances.png')

        evo_model_temp.new = MagicMock(return_value=evo_model)  # bypass new model creation to enable mocking
        psi_init = {'mu_v': 1.0, 'tau_v': 50.0, 'mu_w': 1.0, 'tau_w': 50.0}

        # --------- Run Cellmates EM inference ---------
        em_alg = EM(n_states, evo_model=evo_model_temp, obs_model=obs_model)
        results = []
        for i, (v,w) in enumerate(cell_pairs):
            theta_init = np.array([0.25, 0.25, 0.25])
            pC1_v = testing.get_marginals_from_cnp(cnps[v], n_states)[0]
            pC1_w = testing.get_marginals_from_cnp(cnps[w], n_states)[0]
            evo_model.get_one_slice_marginals = MagicMock(return_value=(pC1_v, pC1_w))
            evo_model._expected_changes = MagicMock(return_value=(D[v,w], Dp[v,w], -1.0))
            res_vw = em_alg._fit_quadruplet(v, w, x, theta_init=theta_init, psi_init=psi_init, max_iter=max_iter, rtol=rtol)
            results.append(res_vw)

        distances = -np.ones((n_cells, n_cells, 3))
        iterations = -np.ones((n_cells, n_cells))
        loglikelihoods = -np.ones((n_cells, n_cells))
        # collect results
        for (u, v), l_i, loglik, it in results:
            distances[u, v, :] = l_i
            iterations[(u, v)] = it
            loglikelihoods[(u, v)] = loglik

        L1_diff = np.zeros((n_cells, n_cells, 3))
        for v,w in cell_pairs:
            L1_diff[v,w,:] = abs(distances[v,w,:] - exp_distances[v, w])

        tot_L1_diff = L1_diff.sum(axis=(0,1))
        print(f"Total L1 diff: \n {tot_L1_diff}")

        # Build tree from inferred distances
        tree_res_nx = neighbor_joining.build_tree(distances)
        tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx, taxon_namespace=tree_dp.taxon_namespace)
        # Compare with standard NJ tree
        pairwise_distances = distances[:, :, 1] + distances[:, :, 2]
        # symmetrize
        pairwise_distances = np.triu(pairwise_distances) + np.triu(pairwise_distances, k=1).T
        np.fill_diagonal(pairwise_distances, 0)
        skbio_dm = skbio.DistanceMatrix(pairwise_distances, ids=[str(i) for i in range(n_cells)])
        tree_nj_skbio = skbio.tree.nj(skbio_dm)
        tree_nj_skbio = tree_nj_skbio.root_at_midpoint()
        tree_nj_dp = dendropy.Tree.get(data=str(tree_nj_skbio), schema="newick",
                                                          taxon_namespace=tree_dp.taxon_namespace)
        tree_utils.label_tree(tree_nj_dp, method='int')
        tree_nj_nx = tree_utils.convert_dendropy_to_networkx(tree_nj_dp)

        # Save the pairwise distance matrix
        fig, ax = plt.subplots()
        visual.plot_cell_pairwise_heatmap(pairwise_distances, label=np.arange(0, n_cells), ax=ax)
        fig.savefig(out_dir + '/inferred_pairwise_distances.png')

        #print(f"Inferred tree:")
        #nx.write_network_text(tree_res_nx)
        #print(f"True tree:")
        #nx.write_network_text(tree_nx)

        # Compare trees
        rf_dist = treecompare.symmetric_difference(tree_dp, tree_res_dp)
        print(f"RF dist: \n {rf_dist}")
        rf_dist_nj = treecompare.symmetric_difference(tree_dp, tree_nj_dp)
        print(f"RF dist_nj: \n {rf_dist_nj}")

        fig, axs = plt.subplots(1, 3, figsize=(15, 10))
        _, ax1 = visual.draw_graph(tree_nx, ax=axs[0])
        _, ax2 = visual.draw_graph(tree_res_nx, ax=axs[1])
        _, ax3 = visual.draw_graph(tree_nj_nx, ax=axs[2])
        axs[0].set_title('True tree')
        axs[1].set_title('Inferred tree')
        axs[2].set_title('NJ tree')
        fig.savefig(out_dir + '/true_inferred_and_NJ_tree.png')

        self.assertLessEqual(rf_dist, rf_dist_nj)

    def test_cellmates_simple_tree(self):
        # Inference parameters
        max_iter = 20
        tol = 1e-4

        # Simulation parameters
        n_sites = 200
        n_cells = 4
        n_states = 5
        n_clonal_events_per_edge = 3
        n_focal_events_per_edge = 0
        clonal_CN_length = 20
        obs_model_sim = 'normal'
        sim_evo_model = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           n_focal_events=n_focal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length)

        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model_sim,
                            evo_model=sim_evo_model)
        x = data['obs']
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)

        out_dir = testing.create_output_test_folder(sub_folder_name=f"Cellmates_EM_M{n_sites}_N{n_cells}_A{n_states}")
        fig, ax = plt.subplots()
        visual.plot_cn_profile(cnps, ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

        # --------- Setup Models ---------
        obs_model = obs_model_sim
        evo_model = JCBModel(n_states=n_states)
        # --------- Run Cellmates EM inference ---------
        em_alg = EM(n_states, evo_model=evo_model, obs_model=obs_model)
        em_alg.fit(x, max_iter=max_iter, tol=tol, num_processors=1)

        distances = em_alg.distances
        print(f"Distance matrix: \n {distances[0, ...]}")

        # Get the inferred tree
        tree_res_nx = neighbor_joining.build_tree(distances, internal_indexing=True)

        nx.write_network_text(tree_nx)
        nx.write_network_text(tree_res_nx)

        tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx, taxon_namespace=tree_dp.taxon_namespace)

        # Compare trees
        rf_dist = treecompare.symmetric_difference(tree_dp, tree_res_dp)
        print(f"RF dist: \n {rf_dist}")

        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        _, ax1 = visual.draw_graph(tree_nx, ax=axs[0])
        _, ax2 = visual.draw_graph(tree_res_nx, ax=axs[1])
        axs[0].set_title('True tree')
        axs[1].set_title('Inferred tree')
        fig.savefig(out_dir + '/true_inferred_and_NJ_tree.png')

    @unittest.skip("Under development")
    def test_dice_benchmark_PoC_data(self):
        # TODO: finish this test
        cnasim_tree_nw = "((leaf1:0.01,leaf2:0.01):0.127,((leaf3:0.039,(leaf5:0.023,(leaf7:0.01,leaf8:0.01):0.013):0.016):0.096,(leaf4:0.12,(leaf6:0.061,(leaf9:0.018,leaf10:0.018):0.043):0.059):0.015):0.003)root"
        cnasim_tree_nx = tree_utils.newick_to_nx(cnasim_tree_nw, interior_node_names=[f"int{i+10}" for i in range(10)])

        # CNPs
        adata = anndata.read_h5ad("../../data/DICE_benchmarks/cnasim_benchmark_A1_0_1.h5ad")
        cnps = adata.X
        out_dir = testing.create_output_test_folder()
        fig, ax = visual.draw_graph(cnasim_tree_nx, save_path=out_dir + '/cnasim_tree.png')
        # save
        fig.savefig(out_dir + '/cnasim_tree.png')
        fig, ax = visual.plot_cn_profile(cnps, ax=ax)
        fig.savefig(out_dir + '/cnasim_cn_profile.png')

        # Reconstruct internal CNPs based on minimal evolution
        cnps_all = tree_utils.reconstruct_internal_cnps(leaf_cnps=..., tree_nx=cnasim_tree_nx, n_states=7, method='min_evolution')

        # Run Cellmates EM inference on the CNPs








