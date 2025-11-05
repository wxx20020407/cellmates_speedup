import unittest

import numpy as np

from cellmates.models.evo import SimulationEvoModel
from cellmates.models.obs import PoissonModel
from cellmates.simulation import datagen
from cellmates.utils import testing


class PoissonModelTestCase(unittest.TestCase):

    def test_sample(self):
        model = PoissonModel(n_states=7, lambda_v_prior=100)
        cn_vw = np.array([
            [2, 2, 3, 3, 3, 3],
            [2, 2, 3, 3, 4, 4]
        ], dtype=int)
        r_vw = model.sample(cn_vw)

        self.assertEqual(r_vw.shape, (6, 2))
        self.assertTrue(np.all(r_vw >= 0))
        self.assertTrue(np.all(r_vw == np.round(r_vw)))
        print(r_vw)

    def test_log_emission(self):
        model = PoissonModel(n_states=7, lambda_v_prior=100)
        cn_vw = np.array([
            [2, 2, 3, 3, 3, 3],
            [2, 2, 3, 3, 4, 4]
        ], dtype=int)
        r_vw = model.sample(cn_vw)
        self.assertEqual(r_vw.shape, (6, 2))
        log_p = model.log_emission(r_vw)  # should have shape (n_sites, n_states, n_states)

        self.assertEqual(log_p.shape, (6, 7, 7))
        self.assertTrue(np.all(log_p <= 0))
        # check that the likelihood over the gt copy number is higher than other states
        loglik_gt = np.sum(log_p[np.arange(6), cn_vw[0], cn_vw[1]])
        loglik_other = np.sum(log_p, axis=0)
        self.assertTrue(np.all(loglik_gt > loglik_other))

    def test_update_given_c(self):
        """
        Test updating the PoissonModel parameters given copy number profiles.
        Currently only works if lambda_w_true = lambda_v_true due to data simulation implementation.
        """
        n_sites = 100
        K = 7
        lambda_v_true = lambda_w_true = 100.0
        obs_model = PoissonModel(n_states=K, lambda_v_prior=lambda_v_true, lambda_w_prior=lambda_w_true, train=True)
        evo_sim_model = SimulationEvoModel(n_clonal_CN_events=5, n_focal_events=5, clonal_CN_length=n_sites // 20)
        # Simulate data
        data = datagen.simulate_quadruplet(n_sites, obs_model, evo_sim_model, n_states=K)
        # Initialize parameters away from true values
        obs_model.lambda_v = lambda_v_true * 0.5
        # Update parameters given true copy numbers
        x = data['obs']
        cnps = data['cn']
        pC1_v, _ = testing.get_marginals_from_cnp(cnps[0], K)
        pC1_w, _ = testing.get_marginals_from_cnp(cnps[1], K)
        obs_model.update(x, (pC1_v, pC1_w))
        updated_psi = obs_model.psi
        self.assertAlmostEqual(updated_psi['lambda_v'], lambda_v_true, delta=lambda_v_true * 0.1)
        self.assertAlmostEqual(updated_psi['lambda_w'], lambda_w_true, delta=lambda_w_true * 0.1)

    def test_nan_values(self):
        """
        Tests that the model handles NaN values in the observations without errors.
        """
        n_sites = 100
        K = 7
        obs_model = PoissonModel(n_states=K, lambda_v_prior=100.)
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=5, n_focal_events=5, clonal_CN_length=10)
        data = datagen.simulate_quadruplet(n_sites, obs_model, evo_model_sim, n_states=K)
        x = data['obs']
        x_full = x.copy()
        # Introduce NaN values randomly
        nan_indices = np.random.choice(n_sites, size=10, replace=False)
        x[nan_indices, 0] = np.nan
        x[nan_indices, 1] = np.nan

        # Attempt to compute log emission probabilities
        try:
            log_p = obs_model.log_emission(x)
            self.assertTrue(not np.any(np.isnan(log_p)), "Log probabilities should not contain NaNs.")
            self.assertEqual(log_p.shape, (n_sites, K, K), "Log probabilities shape mismatch.")
            # sum of log emissions is higher if there are nans
            loglik_full = np.sum(obs_model.log_emission(x_full))
            loglik_nan = np.sum(obs_model.log_emission(x))
            self.assertTrue(loglik_nan > loglik_full - 1e-3,
                            "Log likelihood with NaNs should be higher than or equal to full data.")
            print("Log emission computed successfully with NaN values.")
        except Exception as e:
            self.fail(f"Model failed to handle NaN values: {e}")

        # perform M-step update
        gamma = np.random.rand(n_sites, K)
        gamma /= gamma.sum(axis=1, keepdims=True)
        obs_model.M_step(obs_vw=x[:, [0, 1]], conditionals_vw=[gamma, gamma])
        self.assertTrue(all(not np.isnan(val).any() for val in obs_model.psi.values()))
