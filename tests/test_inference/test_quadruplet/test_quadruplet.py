import unittest
from random import random
from models.observation_models import QuadrupletSpecificCellbaselineAndPrecisionModel
import numpy as np

from models.quadruplet import Quadruplet


class QuadrupletTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)
        # TODO: add logging lever

    def test_quadruplet_data_simulation(self):
        M = 10
        A = 3
        eps_a = 5.0
        eps_b = M * 1.0
        eps_0 = 0.05
        mu_v = 1.0
        mu_w = 1.0
        tau_v = 10.0
        tau_w = 10.0
        C_r = np.ones(M, A) * 2.0
        obs_model = QuadrupletSpecificCellbaselineAndPrecisionModel(M, mu_v, mu_w, tau_v, tau_w)


        quadruplet = Quadruplet(M, A, C_r, yv=None, yw=None, CN_model=None, obs_model=obs_model)

