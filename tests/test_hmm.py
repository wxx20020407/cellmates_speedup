import itertools
import unittest
import numpy as np
import numpy.testing as npt
from scipy import special as sp

from models.evo import JCBModel, CopyTree
from models.evo.basefunc import p_delta
from models.obs import PoissonModel, NormalModel


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

    def test_forward_backward(self):
        # without normalization we have that ll = \sum_{c} \alpha_{t, c} \beta_{t, c} for all t
        np.random.seed(101)
        n_sites = 12
        n_states = 5
        obs_model = NormalModel(n_states=n_states)

        cnp = np.array([
            [2] * n_sites,
            [2] * (n_sites // 2) + [3] * (n_sites // 2),
            [2] * (n_sites // 2) + [3] * (n_sites // 4) + [4] * (n_sites // 4),
            [2] * (n_sites // 2) + [3] * (n_sites // 4) + [1] * (n_sites // 4)
        ])
        # generate data for two cells
        obs = obs_model.sample(cnp[2:4])

        # compute with forward-algorithm with normalized forward variables
        log_emissions = obs_model.log_emission(obs)

        evo_model = CopyTree(n_states=n_states)
        evo_model.theta = np.array([1/12, 1/12, 1/12])

        _, ll_fwd = evo_model._forward_pass_likelihood(obs, log_emissions, normalization=True)
        alpha, _ = evo_model._forward_pass_likelihood(obs, log_emissions, normalization=False)
        beta = evo_model.backward_pass(obs, log_emissions, normalization=False)
        for t in range(n_sites):
            ll = sp.logsumexp(alpha[t] + beta[t])
            # print(f"t={t}, ll: {ll}, ll_fwd: {ll_fwd}")
            self.assertAlmostEqual(ll, ll_fwd)



if __name__ == '__main__':
    unittest.main()
