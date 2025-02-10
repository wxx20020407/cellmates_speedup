import unittest

import numpy as np

from models.obs import PoissonModel


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

