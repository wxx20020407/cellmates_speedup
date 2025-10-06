import unittest
import numpy as np

from cellmates.models.evo import EvoModel, JCBModel
from cellmates.models.obs import NormalModel
from cellmates.utils.math_utils import l_from_p


class TestEvoModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_multi_chr_expected_changes(self):
        n_states = 5
        n_bins = 30
        obs = np.array([
            [2] * 10 +          [3] * 5 + [2] * 5 + [2] * 10,            # chr2p gain (1/30 changes)
            [2] * 5 + [3] * 5 + [2] * 10 +          [4] * 10             # chr1q gain and chr3 dup (2/30 changes)
        ], dtype=np.float32).T
        # this dataset if split into 3 chromosomes has one change in chr2 (cell 1) and one change in chr1 (cell 2)
        # the full chromosome change in chr3 is not counted according to the JCB model
        # The following branch lengths are a rough estimate to get the expected number of changes right
        l_init = l_from_p(np.array([0.001, 0.033, 0.033]), n_states=n_states)

        chromosome_ends = [10, 20]
        evo_model = JCBModel(n_states=n_states, chromosome_ends=chromosome_ends)
        evo_model.theta = np.array(l_init)
        obs_model = NormalModel(n_states=n_states)
        changes = evo_model.multi_chr_expected_changes(obs, obs_model=obs_model)
        print("\nExpected changes, no-changes, log-likelihood (multi-chr):")
        print(changes)
        lik_multi = changes[-1]
        # compare likelihood with single chromosome
        evo_model_single_chr = JCBModel(n_states=n_states)
        evo_model_single_chr.theta = np.array(l_init)
        changes_single_chr = evo_model_single_chr.multi_chr_expected_changes(obs, obs_model=obs_model)

        print("Single chr expected changes, no-changes, log-likelihood:")
        print(changes_single_chr)
        lik_single = changes_single_chr[-1]
        self.assertGreater(lik_multi, lik_single, msg="Likelihood should improve when accounting for chromosome boundaries given the lengths")
        # l_init in single chr should expect more changes
        l_init_single = l_from_p(np.array([0.001, 0.066, 0.1]), n_states=n_states)
        evo_model_single_chr.theta = np.array(l_init_single)
        changes_single_chr = evo_model_single_chr.multi_chr_expected_changes(obs, obs_model=obs_model)
        print("Single chr expected changes, no-changes, log-likelihood (with lengths matching data as a single chromosome):")
        print(changes_single_chr)
        lik_single_fit = changes_single_chr[-1]
        self.assertGreater(lik_single_fit, lik_single, msg="Likelihood should improve when lengths match data (more changes expected at chromosome boundaries")

