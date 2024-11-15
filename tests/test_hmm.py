import itertools
import unittest
import numpy as np
import numpy.testing as npt

from inference.em import compute_l_start_prob, compute_l_trans_mat
from models.evolutionary_models import p_delta


class MyTestCase(unittest.TestCase):
    def test_start_prob(self):
        # n_states == K + 1
        n_states = 7
        l = np.array([0.07, 0.1, 0.03])
        self.assertTrue(np.all(l <= 1 / (n_states-1)))
        self.assertTrue(np.all(l >= 0.))

        start_prob = compute_l_start_prob(l_trip=l, n_states=n_states)
        self.assertTrue(np.isclose(np.sum(start_prob), 1.))

        trans_mat = compute_l_trans_mat(l, n_states)
        npt.assert_allclose(np.sum(trans_mat, axis=(3, 4, 5)), 1.)

    def test_pdelta_jcb(self):
        n_states = 7
        l = .04
        for (i, ii, j) in itertools.product(range(n_states), repeat=3):
            marg = 0.
            for jj in range(n_states):
                marg += p_delta(n_states, l, i, ii, j, jj)
            self.assertAlmostEqual(marg, 1.)


if __name__ == '__main__':
    unittest.main()
