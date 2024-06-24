import unittest
from random import random
from src.models.observation_models.normalized_read_counts_models import QuadrupletSpecificCellbaselineAndPrecisionModel
import numpy as np

from models.quadruplet import Quadruplet


class QuadrupletTestCase(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed=101)

    def test_quadruplet_data_simulation(self):
        M = 10
        A = 5
        eps_a = 5.0
        eps_b = M * 1.0
        eps_0 = 0.05
        mu_v = 1.0
        mu_w = 1.0
        tau_v = 100.0
        tau_w = 100.0
        C_r = np.ones(M) * 2.0
        obs_model = QuadrupletSpecificCellbaselineAndPrecisionModel(M, mu_v, mu_w, tau_v, tau_w)

        quadruplet = Quadruplet(M, A, C_r, yv=None, yw=None, CN_model=None, obs_model=obs_model)
        out = quadruplet.simulate_data(eps_a, eps_b, eps_0, mu_v, mu_w, tau_v, tau_w)

        expected_Y_v = out['c_v'] * out['mu_v']
        expected_Y_w = out['c_w'] * out['mu_w']

        self.assertTrue(np.allclose(out['yv'], expected_Y_v, atol=0.3))
        self.assertTrue(np.allclose(out['yw'], expected_Y_w, atol=0.3))

