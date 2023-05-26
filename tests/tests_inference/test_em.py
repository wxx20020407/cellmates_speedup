import unittest

import numpy as np

from inference.em import EM
from models.quadruplet import Quadruplet

from src.inference.em import em_alg


def _generate_obs():
    # 10 sites, 5 cells
    return np.array([
        [200] * 5 + [300] * 5,
        [100] * 5 + [200] * 5,
        [100] * 3 + [200] * 2 + [300] * 5,
        [200] * 10,
        [400] * 2 + [300] * 2 + [200] * 3 + [100] * 3
    ]), np.ones((5, 5))


class EMTestCase(unittest.TestCase):

    def test_simple_hmm_hmmlearn(self):
        M = 20
        A = 5
        C_r = np.zeros((M, A))
        C_r[:, 2] = 1
        y1 = []
        y2 = []
        for m in range(M):
            y1.append([np.random.poisson(1.0)])
            y2.append([np.random.poisson(1.0)])

        obs_1 = np.ones(M, ) * 2.
        obs_2 = np.ones(M, ) * 2.
        quad = Quadruplet(M, A, C_r, y1, y2)
        em = EM(quad)
        em.run_hmmlearn()  # Throws exception from hmmlearn expecting 1D hidden state

    def test_em_alg(self):
        # generate toy data
        obs, eps = _generate_obs()
        # run em
        ctr_table = em_alg(obs)
        # assert epsilons
        print(ctr_table)
