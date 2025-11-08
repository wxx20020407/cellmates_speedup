import unittest
import numpy as np

from cellmates.models.obs import JitterCopy


class TestJitterCopy(unittest.TestCase):

    def setUp(self):
        self.n_states = 5
        self.error_rate = 0.1
        self.model = JitterCopy(n_states=self.n_states, error_rate=self.error_rate)
        self.model.initialize()

    def test_sample(self):
        n_cells, n_sites = 3, 4
        cnp = np.random.randint(1, self.n_states, size=(n_cells, n_sites))
        samples = self.model.sample(cnp)

        # Check shape
        self.assertEqual(samples.shape, (n_sites, n_cells))
        # Check integer type
        self.assertTrue(np.issubdtype(samples.dtype, np.integer))
        # Check non-negative
        self.assertTrue(np.all(samples >= 0))

    def test_log_emission(self):
        # Simulated observations for two cells v,w
        obs_vw = np.array([[2, 3],
                           [1, 1],
                           [4, 2],
                           [0, 3]])
        log_em = self.model.log_emission(obs_vw)

        # Shape should be (n_sites, n_states, n_states)
        self.assertEqual(log_em.shape, (obs_vw.shape[0], self.n_states, self.n_states))

        # Log probabilities should be finite
        self.assertTrue(np.all(np.isfinite(log_em)))

        # Probabilities reconstructed from log_emission_split should sum < 1 (since it's per-integer mass)
        log_v, log_w = self.model.log_emission_split(obs_vw)
        self.assertTrue(log_v.shape, (obs_vw.shape[0], self.n_states))
        prob_v = np.exp(log_v)
        prob_w = np.exp(log_w)
        self.assertTrue(np.all(prob_v.sum(axis=1) <= 1.0 + 1e-6))
        self.assertTrue(np.all(prob_w.sum(axis=1) <= 1.0 + 1e-6))

if __name__ == '__main__':
    unittest.main()
