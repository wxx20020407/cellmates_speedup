import logging
import unittest
import os

import anndata as ad
import numpy as np
from Bio import Phylo
import dendropy as dpy
from matplotlib import pyplot as plt

from cellmates.models.evo import SimulationEvoModel
from cellmates.other_methods import dice_api
from cellmates.simulation import datagen
from cellmates.utils import testing, visual, tree_utils


class DiceAPITestCase(unittest.TestCase):

    def setUp(self):
        self.test_out_dir_rel_path = '../../output/tests/test_other_methods/test_dice'
        self.test_data_dir_rel_path = '../testdata/dice'

    def test_run_dice(self):
        logging.basicConfig(level=logging.INFO)
        path_to_testdata = self.test_data_dir_rel_path + "/sampleProfiles.tsv"
        out_dir = self.test_out_dir_rel_path + '/test_run_dice'
         # Run DICE on the test data
        dice_api.run_dice(path_to_testdata, out_path=out_dir)
        self.assertTrue(os.path.exists(out_dir))

    def test_toy_data_and_run_dice(self):
        # Simulate a small AnnData object
        n_cells = 10
        n_bins = 50
        X = np.random.randint(0, 5, size=(n_cells, n_bins))
        states = np.random.randint(0, 3, size=(n_cells, n_bins))
        cnps = np.stack((states, states), axis=-1)  # shape (n_cells, n_bins, 2)

        # Save simulated data tsv-file to temporary directory
        dataset_path = self.test_out_dir_rel_path + '/test_simulate_data_and_run_dice/simulated_data'
        chr_ends_idx = [10, 20, 35, n_bins-1]
        bin_length = 1000
        dice_api.convert_to_dice_tsv(cnps, chr_ends_idx, bin_length, dataset_path + '_states.tsv')

        # Run DICE on the simulated data
        dice_input_path = dataset_path + '_states.tsv'
        dice_out_dir = self.test_out_dir_rel_path + '/test_toy_data_and_run_dice'
        dice_api.run_dice(dice_input_path, dice_out_dir)

    def test_simulate_data_and_run_dice(self):
        """
        Simulates data using the SimulationEvoModel, converts it to DICE format, and runs DICE.
        Then loads the DICE output and checks if it is as expected.
        """
        N, M, K = 5, 500, 7  # number of cells, bins, states
        M_tot = 2*M # total bins for both haplotypes
        n_clonal_events_per_edge = 0
        n_focal_events_per_edge = 3
        clonal_CN_length_ratio = 0.1
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                            n_focal_events=n_focal_events_per_edge,
                                            clonal_CN_length_ratio=clonal_CN_length_ratio)
        out_sim_hap_a = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N)
        true_tree = out_sim_hap_a['tree']
        out_sim_hap_b = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N, tree=true_tree)
        cnps_hap_a = out_sim_hap_a['cn']
        cnps_hap_b = out_sim_hap_b['cn']
        cnps = np.stack((cnps_hap_a, cnps_hap_b), axis=-1)  # shape (2*n_cells-1, n_bins, 2)
        cnps_obs = cnps[:N, :, :] # take only leaf nodes (n_cells, n_bins, 2)

        # Save simulated data for visualization
        out_dir = testing.create_output_test_folder(sub_folder_name=f"N_{N}_M{M}_K{K}_CN{n_clonal_events_per_edge}_fCN{n_focal_events_per_edge}")
        fig, ax = plt.subplots(2, 1)
        visual.plot_cn_profile(cnps_hap_a, cell_labels=np.arange(0, N), ax=ax[0], title="Hap A")
        visual.plot_cn_profile(cnps_hap_b, cell_labels=np.arange(0, N), ax=ax[1], title="Hap B")
        fig.savefig(out_dir + '/cn_profile.png')

        # Save simulated data tsv-file to temporary directory
        dataset_path = out_dir + '/simulated_data'
        chr_ends_idx = [M//2, M-1]
        bin_length = 1000
        dice_api.convert_to_dice_tsv(cnps_obs, chr_ends_idx, bin_length, dataset_path + '_states.tsv')

        # Run DICE on the simulated data
        out_dir_rel_test_dir = out_dir.partition('/test_dice')[2]
        dice_out_dir = self.test_out_dir_rel_path + out_dir_rel_test_dir
        dice_input_path = dice_out_dir + '/simulated_data_states.tsv'
        dice_api.run_dice(dice_input_path, dice_out_dir)

        # Load and check DICE output
        # Load DICE tree
        cell_names = ['cell_'+str(i) for i in range(N)]
        taxon_namespace = true_tree.taxon_namespace # dpy.TaxonNamespace(cell_names)
        dice_nwk_file_path = dice_out_dir + '/standard_root_balME_tree.nwk'
        newick_str = open(dice_nwk_file_path).read().strip()
        dice_tree_bio = Phylo.read(dice_nwk_file_path, 'newick')
        dice_tree_dpy = dpy.Tree.get(data=newick_str, schema='newick', taxon_namespace=taxon_namespace)

        rf_dist_dice_vs_true = tree_utils.normalized_rf_distance(true_tree, dice_tree_dpy)

        true_tree.print_plot()
        dice_tree_dpy.print_plot()


        print(f"Normalized RF distance DICE vs true tree: {rf_dist_dice_vs_true}")



