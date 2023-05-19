import unittest

import numpy as np

from inference.em import EM


class EMTestCase(unittest.TestCase):

    def test_simple_hmm(self):
        M = 10
        A = 5
        C_r = np.zeros((M, A))
        C_r[:, 2] = 1
        obs_1 = np.ones(M,) * 2.
        obs_2 = np.ones(M,) * 2.
        em = EM(M, A, C_r, obs_1, obs_2)
        em.run()