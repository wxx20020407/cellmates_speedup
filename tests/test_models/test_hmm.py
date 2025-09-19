import itertools
import logging
import unittest
import numpy as np
import numpy.testing as npt
from scipy import special as sp

from cellmates.inference.em import EM
from cellmates.models.evo import JCBModel, CopyTree
from cellmates.models.evo.basefunc import p_delta
from cellmates.models.obs import PoissonModel, NormalModel
from cellmates.utils.math_utils import compute_cn_changes


class HMMTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(101)

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

    def test_forward_algorithm(self):
        n_sites = 12
        n_states = 5
        evo_model = CopyTree(n_states=n_states)
        evo_model.eps = np.array([1/12, 1/12, 1/12])
        obs_model = PoissonModel(n_states=n_states)

        # copy number profiles
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
        alpha, ll_fwd = evo_model._forward_pass_likelihood(obs, log_emissions)
        print(f"alpha: {sp.logsumexp(alpha, axis=(1, 2, 3))}")
        print(f"loglik fwd norm: {ll_fwd}")


        # compute with backward-algorithm with normalized backward variables

        # compute "by hand"
        alphaM = np.zeros((n_states, n_states, n_states))
        alphaM[...] = np.log(evo_model.start_prob) + log_emissions[0, None, ...]
        # print(f"alpha1(3, 3, 3): {alphaM[3, 3, 3]}")
        # print(f"alpha1_sum: {logsumexp(alphaM)}")
        for m in range(1, n_sites):
            alphaM[...] = log_emissions[m, None, :, :] + sp.logsumexp(alphaM[:, :, :, None, None, None] + np.log(evo_model.trans_mat), axis=(0, 1, 2))
            # print(f"alpha{m+1}(3, 3, 3): {alphaM[3, 3, 3]}")
            # print(f"alpha{m+1}_sum: {logsumexp(alphaM)}")

        ll = sp.logsumexp(alphaM)
        print(f"by hand ll {ll}")

        self.assertAlmostEqual(ll_fwd, ll)

    def test_trivial_likelihood(self):
        logging.basicConfig(level=logging.DEBUG)

        cn = np.array([
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [1, 0, 1, 2],
            [1, 1, 3, 3],
        ])
        print("cn\n", cn)

        n_states = 4
        n_sites = cn.shape[1]
        evo_model = CopyTree(n_states=n_states)
        # obs_model = PoissonModel(n_states=n_states, lambda_v_prior=10)
        obs_model = NormalModel(n_states, mu_v_prior=1., tau_v_prior=50, mu_w_prior=1., tau_w_prior=50)
        obs = obs_model.sample(cn[2:4])
        print("variance", 1/obs_model.tau_v_prior)
        print("obs.T\n", obs.transpose())
        log_emissions = obs_model.log_emission(obs)
        log_emissions_cn_estimate = np.stack(np.unravel_index(np.argmax(log_emissions.reshape(-1, n_states**2), axis=1), shape=(n_states, n_states)))
        print("logemission argmax:\n", log_emissions_cn_estimate)

        self.assertTrue(np.all(log_emissions_cn_estimate == cn[2:4]), "obs are too noisy for the purpose of this test")

        cn_changes_ratio = np.array(compute_cn_changes(cn, [(0, 1), (1, 2), (1, 3)])) / n_sites
        print("computed cn changes:\n", cn_changes_ratio)

        em = EM(n_states, obs_model, evo_model, tree_build='ctr', verbose=2)
        em.fit(obs, max_iter=100, num_processors=1, rtol=1e-10, theta_init=np.array([0.2, 0.5, 0.2]))

        print("estimated cn changes:\n", em.distances[0, 1])

        # compute brute force likelihood
        ll_brute_force = -np.inf
        evo_model.eps = cn_changes_ratio
        for i1, j1, k1, i2, j2, k2, i3, j3, k3, i4, j4, k4 in itertools.product(range(n_states), repeat=12):
            # start prob * emission1 * trans prob(1->2) * emission2 * trans prob(2->3) * emission3 * trans prob(3->4) * emission4
            ll_brute_force = np.logaddexp(ll_brute_force, np.log(evo_model.start_prob[i1, j1, k1]) + log_emissions[0, j1, k1] +
                np.log(evo_model.trans_mat[i1, j1, k1, i2, j2, k2]) + log_emissions[1, j2, k2] +
                np.log(evo_model.trans_mat[i2, j2, k2, i3, j3, k3]) + log_emissions[2, j3, k3] +
                np.log(evo_model.trans_mat[i3, j3, k3, i4, j4, k4]) + log_emissions[3, j4, k4])
        print("ll_brute_force", ll_brute_force)

        ll_fwbw = em.compute_pair_likelihood(obs, theta=cn_changes_ratio)
        print("ll_fwbw", ll_fwbw)
        self.assertAlmostEqual(ll_brute_force, ll_fwbw, places=5, msg="likelihood computation with forward backward is"
                                                                      " different than bruteforce, hence `most likely`"
                                                                      " bugged")




if __name__ == '__main__':
    unittest.main()
