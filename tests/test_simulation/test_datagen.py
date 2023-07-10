import random
import unittest
import numpy as np

from simulation.datagen import rand_dataset


class MyTestCase(unittest.TestCase):
    def test_rand_dataset(self):
        random.seed(1234)
        np.random.seed(1234)
        n_cells = 50
        n_states = 7
        n_sites = 200

        data = rand_dataset(n_cells, n_states, n_sites, obs_type='pois')

        self.assertEqual(data['obs'].shape, (n_sites, n_cells))
        self.assertEqual(data['cn'].shape, (2 * n_cells - 1, n_sites))
        for i, t in enumerate(data['tree'].leaf_node_iter()):
            self.assertEqual(data['tax_id_map'][t.taxon], i)


if __name__ == '__main__':
    unittest.main()
