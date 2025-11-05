import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from cellmates.inference.em import EM
from cellmates.models.evo import SimulationEvoModel, JCBModel
from cellmates.models.obs import NormalModel
from cellmates.simulation import datagen
from cellmates.utils import testing, tree_utils


class NormalModelTestCase(unittest.TestCase):

    def test_sample(self):

        model = NormalModel(n_states=7, tau_v_prior=40, tau_w_prior=40)
        cn_vw = np.array([
            [2, 2, 3, 3, 3, 3],
            [2, 2, 3, 3, 4, 4]
        ], dtype=int)
        r_vw = model.sample(cn_vw)
        self.assertEqual(r_vw.shape, (6, 2))
        self.assertTrue(np.all(r_vw >= 0))
        self.assertTrue(np.all(cn_vw.T == np.round(r_vw, 0)))

    def test_log_emission(self):
        model = NormalModel(n_states=7, mu_v_prior=100, tau_v_prior=1)
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
        Tests that the NormalModel parameters are updated correctly given true copy number profiles.
        """
        n_sites = 1000
        K = 7
        mu_v_true = mu_w_true = 1.0
        tau_v_true = tau_w_true = 50.0
        obs_model = NormalModel(n_states=K, mu_v_prior=mu_v_true, tau_v_prior=tau_v_true, train=True)
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=5, n_focal_events=5, clonal_CN_length=n_sites // 20)
        # Simulate data
        data = datagen.simulate_quadruplet(n_sites, obs_model, evo_model_sim, n_states=K)
        # Initialize parameters away from true values
        psi_init = {'mu_v': mu_v_true * 4, 'tau_v': tau_v_true * 4, 'mu_w': mu_w_true * 2, 'tau_w': tau_w_true * 4}
        obs_model.initialize(psi_init)
        print(f"Initial psi: {obs_model.psi}")
        # Update parameters given true copy numbers
        x = data['obs']
        cnps = data['cn']
        pC1_v, _ = testing.get_marginals_from_cnp(cnps[0], K)
        pC1_w, _ = testing.get_marginals_from_cnp(cnps[1], K)
        obs_model.update(x, (pC1_v, pC1_w))
        updated_psi = obs_model.psi
        print(f"Updated psi: {[updated_psi[key].round(2).item() for key in updated_psi.keys()]}")
        print(f"Expected psi: mu_v={mu_v_true}, tau_v={tau_v_true}, mu_w={mu_w_true}, tau_w={tau_w_true}")
        self.assertAlmostEqual(updated_psi['mu_v'], mu_v_true, delta=0.1)
        self.assertAlmostEqual(updated_psi['mu_w'], mu_w_true, delta=0.1)
        self.assertAlmostEqual(updated_psi['tau_v'], tau_v_true, delta=10.0)
        self.assertAlmostEqual(updated_psi['tau_w'], tau_w_true, delta=10.0)

    def test_EM_given_c(self):
        """
        Tests that the update step works correctly when given the true copy numbers.
        Initializes the psi parameters away from the true values and checks that they are updated towards them.
        """
        n_sites = 100
        K = 7
        mu_v_true = mu_w_true = 1.0
        tau_v_true = tau_w_true = 50.0
        obs_model = NormalModel(n_states=K, mu_v_prior=mu_v_true, tau_v_prior=tau_v_true, train=True)
        # Simulation data uses obs_model priors as true params by default
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=5, n_focal_events=5, clonal_CN_length=10)
        data = datagen.simulate_quadruplet(n_sites, obs_model, evo_model_sim, n_states=K)
        cnps = data['cn']
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)
        evo_model_temp = JCBModel(n_states=K)
        evo_model = JCBModel(n_states=K)

        # Run several update steps with ideal expected changes
        cell_pairs = [(0, 1)]
        D, Dp = testing.get_expected_changes(cnps, tree_nx, cell_pairs)
        pC1_v, pC2_v = testing.get_marginals_from_cnp(cnps[0], K)
        pC1_w, pC2_w = testing.get_marginals_from_cnp(cnps[1], K)
        assert (np.argmax(pC1_v, axis=1) == cnps[0]).all()
        assert (np.argmax(pC1_w, axis=1) == cnps[1]).all()
        evo_model_temp.new = MagicMock(return_value=evo_model)
        evo_model.get_one_slice_marginals = MagicMock(return_value=[pC1_v, pC1_w])
        # last three values are placeholders for loglikelihood, expected counts and log_gamma
        evo_model._expected_changes = MagicMock(return_value=[D[(0,1)], Dp[(0,1)], -1.0, -1., -1.])

        theta_init = np.array([0.2, 0.2, 0.2])
        psi_init = {'mu_v': mu_v_true*4, 'tau_v': tau_v_true*2, 'mu_w': mu_w_true*2, 'tau_w': tau_w_true*2}

        em = EM(n_states=K, obs_model=obs_model, evo_model=evo_model_temp)
        em.fit(data['obs'], theta_init=theta_init, psi_init=psi_init)

        updated_psi = em.obs_model.psi
        print(f"\n Initial psi: {psi_init}, \n Updated psi: {updated_psi}")
        self.assertAlmostEqual(updated_psi['mu_v'], mu_v_true, delta=0.1)
        self.assertAlmostEqual(updated_psi['mu_w'], mu_w_true, delta=0.1)
        self.assertAlmostEqual(updated_psi['tau_v'], tau_v_true, delta=1.0)
        self.assertAlmostEqual(updated_psi['tau_w'], tau_w_true, delta=1.0)

    def test_nan_values(self):
        """
        Tests that the model handles NaN values in the observations without errors.
        """
        n_sites = 100
        K = 7
        obs_model = NormalModel(n_states=K, mu_v_prior=1.0, tau_v_prior=50.0, train=True)
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


