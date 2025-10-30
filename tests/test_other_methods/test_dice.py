import logging
import unittest
import os
import anndata as ad
import numpy as np

from cellmates.other_methods import dice_api
from cellmates.utils import testing


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

    def test_simulate_data_and_run_dice(self):
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
        dice_out_dir = self.test_out_dir_rel_path + '/test_simulate_data_and_run_dice'
        dice_api.run_dice(dice_input_path, dice_out_dir)
