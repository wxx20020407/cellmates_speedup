import unittest
import numpy as np
import scipy.special as sp

from cellmates.models.evo import EvoModel, JCBModel
from cellmates.models.obs import NormalModel
from cellmates.utils.math_utils import l_from_p


class TestEvoModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_multi_chr_expected_changes(self):
        """
        Test that the multi-chromosome expected changes calculation works as expected.
        """
        n_states = 5
        n_bins = 30
        hmm_alg = 'pomegranate'
        obs = np.array([
            [2] * 3 + [3] * 7 + [2] * 3 + [3] * 2 + [2] * 5 + [2] * 10,                    # 2/30 changes in multi-chr, 4/30 in single-chr
            [2] * 5 + [3] * 5 + [2] * 5 + [3] * 5           + [2] * 2 + [4] * 2 + [3] * 6  # 4/30 changes in multi-chr, 6/30 in single-chr
        ], dtype=np.float32).T
        # this dataset if split into 3 chromosomes has one change in chr2 (cell 1) and one change in chr1 (cell 2)
        # the full chromosome change in chr3 is not counted according to the JCB model
        # The following branch lengths are a rough estimate to get the expected number of changes right
        l_init = l_from_p(np.array([0.001, 0.067, 0.13]), n_states=n_states)

        chromosome_ends = [10, 20]
        n_chromosomes = len(chromosome_ends) + 1
        evo_model = JCBModel(n_states=n_states, chromosome_ends=chromosome_ends, hmm_alg=hmm_alg)
        evo_model.theta = np.array(l_init)
        obs_model = NormalModel(n_states=n_states)
        changes = evo_model.multi_chr_expected_changes(obs, obs_model=obs_model)
        assert np.isclose(expcounts:=evo_model.expected_counts.sum(), n_bins - n_chromosomes), f"Expected counts should be {n_bins - n_chromosomes}, got {expcounts}"
        print("\nExpected changes, no-changes, log-likelihood (multi-chr):")
        print(changes)
        lik_multi = changes[-1]
        # compare likelihood with single chromosome
        evo_model_single_chr = JCBModel(n_states=n_states, hmm_alg=hmm_alg)
        evo_model_single_chr.theta = np.array(l_init)
        changes_single_chr = evo_model_single_chr.multi_chr_expected_changes(obs, obs_model=obs_model)
        assert np.isclose(expcounts_single:=evo_model_single_chr.expected_counts.sum(), n_bins - 1), f"Expected counts in single chr should be {n_bins - 1}, got {expcounts_single}"

        print("Single chr expected changes, no-changes, log-likelihood:")
        print(changes_single_chr)
        lik_single = changes_single_chr[-1]
        self.assertGreater(lik_multi, lik_single, msg="Likelihood should improve when accounting for chromosome boundaries given the lengths")
        # l_init in single chr should expect more changes
        l_init_single = l_from_p(np.array([0.001, 0.13, 0.2]), n_states=n_states) # slightly longer branches to expect more changes
        evo_model_single_chr.theta = np.array(l_init_single)
        changes_single_chr = evo_model_single_chr.multi_chr_expected_changes(obs, obs_model=obs_model)
        print("Single chr expected changes, no-changes, log-likelihood (with lengths matching data as a single chromosome):")
        print(changes_single_chr)
        lik_single_fit = changes_single_chr[-1]
        self.assertGreater(lik_single_fit, lik_single, msg="Likelihood should improve when lengths match data (more changes expected at chromosome boundaries")

    def test_forward_backward_algs(self):
        """
        Check that the forward-backward implementation using pomegranate and the broadcast implementation give the same results.
        """
        n_states = 4
        n_bins = 20
        obs = np.array([
            [2] * 5 + [3] * 5 + [2] * 10,
            [2] * 10 + [4] * 5 + [2] * 5
        ], dtype=np.float32).T

        evo_model = JCBModel(n_states=n_states, hmm_alg='pomegranate', debug=True)
        evo_model.theta = np.array([0.05, 0.05, 0.05])
        obs_model = NormalModel(n_states=n_states)

        fb_pomegranate = evo_model.forward_backward(obs, obs_model=obs_model)

        evo_model.hmm_alg = 'broadcast'
        fb_broadcast = evo_model.forward_backward(obs, obs_model=obs_model)
        print("\nForward-backward results comparison:")
        print("Expected counts (pomegranate):")
        print(fb_pomegranate[0].sum(axis=(0, 1, 2)))
        print("Expected counts (broadcast):")
        print(fb_broadcast[0].sum(axis=(0, 1, 2)))
        assert np.allclose(fb_pomegranate[0], fb_broadcast[0]), "Expected counts do not match between pomegranate and broadcast implementations"
        print("Gamma (pomegranate):")
        print(sp.logsumexp(fb_pomegranate[1], axis=(1, 2, 3)))
        print("Gamma (broadcast):")
        print(sp.logsumexp(fb_broadcast[1],axis=(1, 2, 3)))
        assert np.allclose(fb_pomegranate[1], fb_broadcast[1]), "Gamma do not match between pomegranate and broadcast implementations"




