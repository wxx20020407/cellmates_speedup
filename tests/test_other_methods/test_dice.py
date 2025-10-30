import logging
import unittest
import os
import anndata as ad
import numpy as np

from cellmates.other_methods import dice_api
from cellmates.utils import testing


class DiceAPITestCase(unittest.TestCase):

    def test_run_dice(self):
        logging.basicConfig(level=logging.INFO)
        path_to_testdata     = '../testdata/dice/sampleProfiles.tsv'
        out_dir = testing.create_output_test_folder()
        dice_api.run_dice(path_to_testdata, out_path=out_dir)
        self.assertTrue(os.path.exists(out_dir))

    def test_simulate_data_and_run_dice(self):
        # Simulate a small AnnData object
        n_cells = 10
        n_bins = 5
        X = np.random.randint(0, 5, size=(n_cells, n_bins))
        states = np.random.randint(0, 3, size=(n_cells, n_bins))

        adata = ad.AnnData(X=X)
        adata.layers['state'] = states

        output_prefix = 'test_dice_output'
        dice_api.anndata_to_dice_tsv(adata, output_prefix)

        # Run DICE on the simulated data
        dice_api.run_dice(f'{output_prefix}_obs.tsv', f'{output_prefix}_states.tsv', out_path='.')
