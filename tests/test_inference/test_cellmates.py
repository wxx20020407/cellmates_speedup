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
from cellmates.models.obs import NormalModel
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

    def test_dice_benchmark_PoC_data_on_expected_lengths(self):
        datasets = ["D1_0"]#, "D2_0", "D3_0", "D4_0", "D5_0", "D6_0", "D7_0", "D8_0"]
        dataseeds = [0]#, 1, 2]
        path_to_data = "../../data/CNAsim/results"
        n_datasets = len(datasets)
        n_seeds = len(dataseeds)
        rf_dist_matrix = np.zeros((n_datasets, n_seeds, 3))

        for i, dataset in enumerate(datasets):
            for dataseed in dataseeds:
                dataset_name = f"{dataset}/{dataseed}"
                print(f"Running test on dataset: {dataset_name}")

                # Load CNASim anndata
                adata = anndata.read_h5ad(os.path.join(path_to_data, dataset_name, "anndata.h5ad"))

                out_dir = testing.create_output_test_folder(sub_folder_name=f"{dataset_name}")
                out_prep = self.prepare_cnasim_data(adata, dataset_name, out_dir, K_max=10)
                cnp_tot, cnp_hap = out_prep['cnp_tot'], out_prep['cnp_hap']
                cnasim_tree_dp, cnasim_tree_nx = out_prep['cnasim_tree_dp'], out_prep['cnasim_tree_nx']
                n_cells = out_prep['n_cells']
                cell_names = out_prep['cell_names']

                # --------- Setup Models and Mock Expected changes D, D' ---------
                n_states = 7
                max_iter = 20
                tol = 1e-4
                # Use CNPs from CNASim construct D, D' matrices
                tree_dp, dist_matrix = tree_utils.make_gt_tree_dist(ad=adata, n_states=n_states, cell_names=cell_names)
                cell_pairs = list(itertools.combinations(range(n_cells), r=2))
                D, Dp = testing.get_expected_changes(cnp_hap, cnasim_tree_nx, cell_pairs)
                l_quad_exp, l_pair_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

                distances = -np.ones((n_cells, n_cells, 3))
                for v, w in cell_pairs:
                    distances[v, w, :] = l_quad_exp[v, w]
                tree_res_nx = neighbor_joining.build_tree(distances, internal_indexing=True)
                tree_res_nx2 = neighbor_joining.build_tree(dist_matrix, internal_indexing=True)
                tree_nj_skbio = tree_utils.skbio_neighbour_joining_from_pairwise_distances(pairwise_distances=l_pair_exp)

                tree_nj_dp = dendropy.Tree.get(data=str(tree_nj_skbio), schema="newick",
                                               taxon_namespace=cnasim_tree_dp.taxon_namespace)
                tree_utils.label_tree(tree_nj_dp, method='int')

                tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx,
                                                                      taxon_namespace=cnasim_tree_dp.taxon_namespace)
                tree_res_dp2 = tree_utils.convert_networkx_to_dendropy(tree_res_nx2,
                                                                       taxon_namespace=cnasim_tree_dp.taxon_namespace)

                # Compare trees
                rf_dist = tree_utils.normalized_rf_distance(cnasim_tree_dp, tree_res_dp)
                rf_dist2 = tree_utils.normalized_rf_distance(cnasim_tree_dp, tree_res_dp2)
                print(f"RF dist: \n {rf_dist}")
                print(f"RF dist2: \n {rf_dist2}")
                rf_dist_nj = tree_utils.normalized_rf_distance(cnasim_tree_dp, tree_nj_dp)
                print(f"RF dist_nj: \n {rf_dist_nj}")

                # Save metrics in numpy array
                rf_out = np.array([rf_dist, rf_dist2, rf_dist_nj])
                np.save(os.path.join(out_dir, 'rf_distances.npy'), rf_out)
                rf_dist_matrix[i, dataseed, :] = rf_out

        # Save summary of all results
        avg_dataset_rf = rf_dist_matrix.mean(axis=1)
        avg_all = rf_dist_matrix.mean(axis=(0,1))
        print(f"Average RF distances per dataset:\n {avg_dataset_rf}")
        print(f"Average RF distances overall:\n {avg_all}")
        np.save(os.path.join(out_dir, 'avg_dataset_rf.npy'), avg_dataset_rf)
        np.save(os.path.join(out_dir, 'avg_all_rf.npy'), avg_all)

        for i in range(n_datasets):
            self.assertLessEqual(avg_dataset_rf[i, 0], avg_dataset_rf[i, 2])  # Centroid based should be better than NJ

    def test_dice_benchmark_PoC_data(self):
        # Load CNASim anndata
        path_to_data = "../../data/CNAsim/results"
        dataset_name = "B5_0/0"
        adata = anndata.read_h5ad(os.path.join(path_to_data, dataset_name, "anndata.h5ad"))

        cnasim_tree_dp, cnasim_tree_nx, cnp_concat, n_cells = self.prepare_cnasim_data(adata, dataset_name)

        # --------- Setup Models and Mock Expected changes D, D' ---------
        n_states = 7
        max_iter = 20
        tol = 1e-4
        # Use CNPs from CNASim construct D, D' matrices
        cell_pairs = list(itertools.combinations(range(n_cells), r=2))
        D, Dp = testing.get_expected_changes(cnp_concat, cnasim_tree_nx, cell_pairs)
        l_quad_exp, l_pair_exp = testing.get_expected_distances(D, Dp, n_states, cell_pairs)

        evo_model_temp = JCBModel(n_states=7)
        evo_model = JCBModel(n_states=7)
        evo_model_temp.new = MagicMock(return_value=evo_model)  # bypass new model
        obs_model = NormalModel(n_states=7, mu_v=1.0, tau_v=10.0, mu_w=1.0, tau_w=10.0)

        psi_init = {'mu_v': 1.0, 'tau_v': 10.0, 'mu_w': 1.0, 'tau_w': 10.0}

        em_alg = EM(n_states, evo_model=evo_model_temp, obs_model=obs_model)

        # Run Cellmates EM inference
        distances = -np.ones((n_cells, n_cells, 3))
        for i, (v,w) in enumerate(cell_pairs):
            theta_init = np.array([0.25, 0.25, 0.25])
            pC1_v = testing.get_marginals_from_cnp(cnp_concat[v], n_states)[0]
            pC1_w = testing.get_marginals_from_cnp(cnp_concat[w], n_states)[0]
            evo_model.get_one_slice_marginals = MagicMock(return_value=(pC1_v, pC1_w))
            evo_model._expected_changes = MagicMock(return_value=(D[v,w], Dp[v,w], -1.0))
            out_quad = em_alg._fit_quadruplet(v, w, cnp_concat, theta_init=theta_init, psi_init=psi_init,
                                  max_iter=max_iter, rtol=tol)
            (v, w), theta_vw, loglik, it = out_quad
            distances[v, w, :] = theta_vw

        tree_res_nx = neighbor_joining.build_tree(distances, internal_indexing=True)

        tree_res_dp = tree_utils.convert_networkx_to_dendropy(tree_res_nx, taxon_namespace=cnasim_tree_dp.taxon_namespace)

        # Compare trees
        rf_dist = treecompare.symmetric_difference(cnasim_tree_dp, tree_res_dp)
        print(f"RF dist: \n {rf_dist}")

    def prepare_cnasim_data(self, adata, dataset_name, out_dir, K_max=10):
        # Get observed CNPs and meta data
        cnp_obs = adata.layers['state']
        cnp_obs_A = adata.layers['Astate']
        cnp_obs_B = adata.layers['Bstate']
        cnp_obs_concat = np.concatenate([cnp_obs_A, cnp_obs_B], axis=1)
        n_cells = cnp_obs.shape[0]
        n_sites = cnp_obs.shape[1]
        C_max = cnp_obs.max()
        n_states = min(C_max, K_max)

        anc_names = adata.uns['ancestral-names']
        anc_mapping = {name: i for i, name in enumerate(anc_names)}
        anc_mapping_tree = {name: n_cells+i for i, name in enumerate(anc_names)}

        # Get true tree
        cnasim_tree_nw = adata.uns['cell-tree-newick']
        cnasim_tree_nx = tree_utils.newick_to_nx(cnasim_tree_nw)
        cell_names = [f"cell{i}" for i in range(1, n_cells + 1)]
        cnasim_tree_nx = tree_utils.relabel_name_to_int(cnasim_tree_nx, cell_names, anc_mapping_tree)
        cnasim_tree_dp = tree_utils.convert_networkx_to_dendropy(cnasim_tree_nx)

        # Get ancestral CNPs
        cnp_anc = adata.uns['ancestral-cn']
        cnp_anc_sorted = cnp_anc[[anc_mapping[name] for name in anc_names], :]
        cnp_tot_concat = np.concatenate([cnp_obs, cnp_anc_sorted], axis=0)

        cnp_anc_A = adata.uns['ancestral-cnA']
        cnp_anc_B = adata.uns['ancestral-cnB']
        cnp_anc_concat = np.concatenate([cnp_anc_A, cnp_anc_B], axis=1)
        cnp_anc_concat_sorted = cnp_anc_concat[[anc_mapping[name] for name in anc_names], :]
        cnp_hap_concat = np.concatenate([cnp_obs_concat, cnp_anc_concat_sorted], axis=0)
        n_ancestors = cnp_anc.shape[0]

        print(f"Number of cells: {n_cells}, number of ancestors: {n_ancestors}, number of sites: {n_sites},"
              f" max CN state: {C_max}, n_states for inference: {n_states}")

        # save visualizations
        fig, ax = visual.draw_graph(cnasim_tree_nx, save_path=out_dir + '/cnasim_tree.png',
                                    node_size=30, with_labels=True)
        fig.savefig(out_dir + '/cnasim_tree.png')
        plt.close()
        fig, ax = plt.subplots()
        ax = visual.plot_cn_profile(cnp_hap_concat, ax=ax)
        fig.savefig(out_dir + '/cnasim_haplotype_cn_profile.png')
        plt.close()
        fig, ax = plt.subplots()
        ax = visual.plot_cn_profile(cnp_tot_concat, ax=ax)
        fig.savefig(out_dir + '/cnasim_total_cn_profile.png')
        plt.close()
        out_dict = {
            'cnp_hap': cnp_hap_concat,
            'cnp_tot': cnp_tot_concat,
            'cnasim_tree_dp': cnasim_tree_dp,
            'cnasim_tree_nx': cnasim_tree_nx,
            'n_cells': n_cells,
            'cell_names': cell_names
        }
        return out_dict









