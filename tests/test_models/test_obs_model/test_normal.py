import time
import unittest

import numpy as np

from models.obs import NormalModel


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

    def test_fast_log_emission(self):
        n_states = 8
        n_sites = 200
        model = NormalModel(n_states=n_states, lambda_v_prior=100)
        cn_vw = np.array([
            [2] * n_sites,
            [2] * n_sites
        ], dtype=int)
        r_vw = model.sample(cn_vw)
        self.assertEqual(r_vw.shape, (n_sites, 2))
        # time
        start = time.time()
        _ = model.log_emission(r_vw)  # should have shape (n_sites, n_states, n_states)
        poiss_t = time.time() - start

        self.assertEqual(r_vw.shape, (n_sites, 2))
        # time
        start = time.time()
        _ = model.log_emission_legacy(r_vw)  # should have shape (n_sites, n_states, n_states)
        norm_t = time.time() - start

        print(f"For-loops time: {poiss_t}, vectorized time: {norm_t}")
        self.assertTrue(poiss_t < norm_t)
