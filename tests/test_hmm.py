import itertools
import unittest
import numpy as np
import numpy.testing as npt

from models.evolutionary_models import p_delta
from models.evolutionary_models.jukes_cantor_breakpoint import JCBModel


class HMMTestCase(unittest.TestCase):
    def test_start_prob(self):
        # n_states == K + 1
        n_states = 7
        l = np.array([0.07, 0.1, 0.03])
        self.assertTrue(np.all(l <= 1 / (n_states-1)))
        self.assertTrue(np.all(l >= 0.))
        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = l

        start_prob = evo_model.start_prob
        self.assertTrue(np.isclose(np.sum(start_prob), 1.))

        trans_mat = evo_model.trans_mat
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
