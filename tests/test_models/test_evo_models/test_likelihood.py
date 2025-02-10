import unittest

import numpy as np
from scipy.special import logsumexp

from models.evo import CopyTree
from models.obs import PoissonModel


class LikelihoodTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(101)

    def test_forward_algorithm(self):
        n_sites = 12
        n_states = 5
        evo_model = CopyTree(n_states=n_states)
        evo_model.eps = np.array([1/12, 1/12, 1/12])
        obs_model = PoissonModel(n_states=n_states)

        # copy number profiles
        cnp = np.array([
            [2] * n_sites,
            [2] * (n_sites // 2) + [3] * (n_sites // 2),
            [2] * (n_sites // 2) + [3] * (n_sites // 4) + [4] * (n_sites // 4),
            [2] * (n_sites // 2) + [3] * (n_sites // 4) + [1] * (n_sites // 4)
        ])
        # generate data for two cells
        obs = obs_model.sample(cnp[2:4])

        # compute with forward-algorithm with normalized forward variables
        log_emissions = obs_model.log_emission(obs)
        alpha, ll_fwd = evo_model._forward_pass_likelihood(obs, log_emissions)
        print(f"alpha: {logsumexp(alpha, axis=(1, 2, 3))}")
        print(f"loglik fwd norm: {ll_fwd}")


        # compute with backward-algorithm with normalized backward variables

        # compute "by hand"
        alphaM = np.zeros((n_states, n_states, n_states))
        alphaM[...] = np.log(evo_model.start_prob) + log_emissions[0, None, ...]
        # print(f"alpha1(3, 3, 3): {alphaM[3, 3, 3]}")
        # print(f"alpha1_sum: {logsumexp(alphaM)}")
        for m in range(1, n_sites):
            alphaM[...] = log_emissions[m, None, :, :] + logsumexp(alphaM[:, :, :, None, None, None] + np.log(evo_model.trans_mat), axis=(0, 1, 2))
            # print(f"alpha{m+1}(3, 3, 3): {alphaM[3, 3, 3]}")
            # print(f"alpha{m+1}_sum: {logsumexp(alphaM)}")

        ll = logsumexp(alphaM)
        print(f"by hand ll {ll}")

        self.assertAlmostEqual(ll_fwd, ll)

