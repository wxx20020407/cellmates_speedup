import itertools
import unittest

import networkx as nx
import numpy as np

from inference.em import EM
from models.quadruplet import Quadruplet

from src.inference.em import em_alg, build_tree


def _generate_obs(noise=0):
    # 10 sites, 5 cells
    obs = np.array([
        [200] * 5 + [300] * 5,
        [100] * 5 + [200] * 5,
        [100] * 3 + [200] * 2 + [300] * 5,
        [200] * 10,
        [400] * 2 + [300] * 2 + [200] * 3 + [100] * 3
    ]).transpose()
    eps = np.ones((5, 5))
    noise = np.round(np.random.normal(size=obs.shape) * noise).astype(int)
    return obs + noise, eps


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
        obs, eps = _generate_obs(noise=10)
        # run em
        ctr_table = em_alg(obs)
        # assert epsilons
        for v, w in itertools.combinations(range(obs.shape[1]), r=2):
            print(f"eps({v},{w}) = {ctr_table[v, w]:.3f}")
            print(np.round((obs[:, [v, w]] / 100)).astype(int).transpose())
            print(" ------- ")
        # print(ctr_table)

    def test_tree_inference(self):
        # generate toy data
        obs, eps = _generate_obs(noise=10)
        # run em
        ctr_table = em_alg(obs)
        # build tree
        em_tree = build_tree(ctr_table)
        print(em_tree)
        assert nx.is_tree(em_tree)

