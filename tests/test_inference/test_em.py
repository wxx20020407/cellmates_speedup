import itertools
import logging
import random
import unittest

import networkx as nx
import numpy as np
from dendropy.calculate import treecompare
from scipy.special import logsumexp

from models.copy_tree import p_delta_change
from simulation.datagen import rand_dataset, get_ctr_table, simulate_cn_seq, emit_raw_obs
from inference.em import jcb_em_alg, compute_exp_changes, two_slice_marginals
from utils.tree_utils import convert_networkx_to_dendropy
from utils.math_utils import l_from_p

from inference.em import em_alg, build_tree


def _generate_obs(noise=0):
    # 10 sites, 5 cells
    obs = np.array([
        [200] * 5 + [300] * 5,
        [100] * 5 + [200] * 5,
        [100] * 3 + [200] * 2 + [300] * 5,
        [200] * 9 + [100],
        [400] * 2 + [300] * 2 + [200] * 3 + [100] * 3
    ]).transpose()
    eps = np.ones((5, 5))
    print("cn:\n")
    print((obs / 100).astype(int).transpose())
    noise = np.round(np.random.normal(size=obs.shape) * noise).astype(int)
    return obs + noise, eps


def get_node2node_distance(tree, node1_label, node2_label):
    tree.calc_node_root_distances()
    node1 = tree.find_node_with_label(node1_label)
    node2 = tree.find_node_with_label(node2_label)
    if node1.root_distance < node2.root_distance:
        node1, node2 = node2, node1
    return node1.root_distance - node2.root_distance


class EMTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)
        # TODO: add logging lever

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

    def test_tree_inference_toy(self):
        # generate toy data
        obs, eps = _generate_obs(noise=10)
        # run em
        # ctr_table = em_alg(obs)
        ctr_table = jcb_em_alg(obs)
        print(ctr_table)
        # build tree
        em_tree = build_tree(ctr_table)
        print(em_tree)
        nx.write_network_text(em_tree, sources=['r'])
        assert nx.is_tree(em_tree)

    def test_tree_inference_synth(self):
        logging.basicConfig(level=logging.WARNING)
        seed = 101
        n_states = 5
        n_sites = 50
        n_cells = 6
        data = rand_dataset(n_cells, n_states, n_sites, alpha=1., obs_type='pois', p_change=0.1, seed=seed)
        print("Generated tree")
        data['tree'].print_plot()
        for node in data['tree'].preorder_node_iter():
            print(node.label, node.edge_length)

        print("Observations")
        print(data['obs'][:20, :])

        ctr_table = jcb_em_alg(data['obs'])
        print(ctr_table)

        em_tree = build_tree(ctr_table)
        # relabel tree nodes with data taxon labels
        nx.write_network_text(em_tree, sources=['r'])

        # compare with true tree using RF-distance (unweighted)
        dendropy_tree = convert_networkx_to_dendropy(em_tree, data['tree'].taxon_namespace)
        dendropy_tree.print_plot()
        rf_distance_jcb = treecompare.symmetric_difference(data['tree'], dendropy_tree)
        print(rf_distance_jcb)

    def test_two_cells(self):
        # seed for reproducibility
        seed = 101
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_cells = 4
        n_states = 5
        n_sites = 1000

        alpha = 1.
        data = rand_dataset(n_cells, n_states, n_sites,
                            alpha=alpha, obs_type='pois', p_change=20 / n_sites, seed=seed)
        print(f"True CTR table")
        true_ctr_table = get_ctr_table(data)
        print(true_ctr_table)
        # for each node print the edge length
        for node in data['tree'].preorder_node_iter():
            print(f"Node {node.label} edge length: {node.edge_length}")
        # tree to newick
        print(data['tree'].as_string('newick'))
        # compare with true tree using RF-distance (unweighted)
        print("--- TRUE TREE ---")
        data['tree'].print_plot(plot_metric='length')

        # pick two leaves whose CTR is not the root
        c1, c2 = 0, 1
        centroid = None
        for r, s in itertools.combinations(range(data['obs'].shape[1]), 2):
            centroid = data['tree'].mrca(taxon_labels=['c' + str(r), 'c' + str(s)])
            if centroid != data['tree'].seed_node:
                c1, c2 = r, s
                break

        print(f"Copy Numbers")
        print(f"R:\t {data['cn'][int(data['tree'].seed_node.label), :]}")
        print(f"U:\t {data['cn'][int(centroid.label), :]}")
        print(f"C{c1}:\t {data['cn'][c1, :]}")
        print(f"C{c2}:\t {data['cn'][c2, :]}")

        print(f"Leaves with non-root CTR: {c1}, {c2}")
        l_uv = get_node2node_distance(data['tree'], centroid.label, c1)
        l_uw = get_node2node_distance(data['tree'], centroid.label, c2)
        print(f"true l_trip: {true_ctr_table[c1, c2]}, {l_uv}, {l_uw}")
        # run EM
        ctr_table = jcb_em_alg(data['obs'][:, [c1, c2]])
        print(f"Estimated CTR table")
        print(ctr_table)
        self.assertAlmostEqual(ctr_table[0, 1], true_ctr_table[c1, c2])
        # build tree
        em_tree = build_tree(ctr_table)

        dendropy_tree = convert_networkx_to_dendropy(em_tree, data['tree'].taxon_namespace)
        print("--- EM TREE ---")
        dendropy_tree.print_plot()

        rf_distance_jcb = treecompare.symmetric_difference(data['tree'], dendropy_tree)
        print(f'Robinson-Fould distance: {rf_distance_jcb}')

    def test_two_slice_marginals(self):
        print(f"\n")
        n_sites = 100
        t0 = n_sites // 6  # first change idx
        t1 = n_sites // 2  # second change idx
        print(f"[2 changes] n_sites: {n_sites}, t0 (1->2): {t0}, t1 (2->3): {t1}")

        n_states = 4
        # reasonable lengths computed by setting the change probability to
        p_change = 3 / n_sites
        ll = l_from_p(p_change * 2, n_states)
        sl = l_from_p(0, n_states)
        print(f"LONG l prop of change: 1 - pdd = {1 - p_delta_change(n_states, ll, change=False)}")
        print(f"SHORT l prop of change: 1 - pdd = {1 - p_delta_change(n_states, sl, change=False)}")
        obs_vw = np.array([
            [100] * (t0 + 1) + [200] * (t1 - t0) + [300] * (n_sites - t1 - 1),
            [100] * (t0 + 1) + [200] * (t1 - t0) + [300] * (n_sites - t1 - 1)
        ]).transpose() + np.random.randint(-10, 10, (n_sites, 2))
        cn_vw = np.round(obs_vw / 100).astype(int)
        print(f"CN VW: {cn_vw.transpose()}")
        n_sites = obs_vw.shape[0]
        # the best explanation is that centroid and root are further apart than centroid and v,u
        # compute two-slice marginals assuming centroid is placed closer to the root
        log_xi_early_centroid = two_slice_marginals(obs_vw, np.array([sl, ll, ll]), n_states, jcb=True)

        # compute two-slice marginals assuming centroid is placed closer to the leaves
        log_xi_late_centroid = two_slice_marginals(obs_vw, np.array([ll, sl, sl]), n_states, jcb=True)

        self.assertEqual(log_xi_late_centroid.shape, (n_sites - 1,) + (n_states,) * 6)
        self.assertTrue(np.allclose(logsumexp(log_xi_late_centroid, axis=(1, 2, 3, 4, 5, 6)), np.zeros(n_sites - 1)))

        # transition from 1 to 2 is more likely in the late scenario due to the higher l_ru
        self.assertGreater(log_xi_late_centroid[t0, 1, 1, 1, 2, 2, 2],
                           log_xi_early_centroid[t0, 1, 1, 1, 2, 2, 2])
        # transition from 1 to 2 is more likely than from 2 to 2 even though l_uv is higher due to observation
        self.assertGreater(log_xi_late_centroid[t0, 1, 1, 1, 2, 2, 2],
                           log_xi_early_centroid[t0, 2, 1, 1, 2, 2, 2])
        # transition from 1 to 2 is maximum in the late scenario
        self.assertEqual(np.unravel_index(np.argmax(log_xi_late_centroid[t0, ...]), log_xi_late_centroid.shape[1:]),
                         (1, 1, 1, 2, 2, 2))
        self.assertEqual(np.unravel_index(np.argmax(log_xi_late_centroid[t1, ...]), log_xi_late_centroid.shape[1:]),
                         (2, 2, 2, 3, 3, 3))
        # we want the transition from m=2 -> m=3 to happen at idx = 2 of log_xi
        joint_prob_log_el = -np.inf
        joint_prob_log_ll = -np.inf
        joint_prob_log_le = -np.inf
        joint_prob_log_ee = -np.inf
        for m in range(n_sites - 1):
            cm1, cm2 = cn_vw[m, 0], cn_vw[m, 1]
            cmm1, cmm2 = cn_vw[m + 1, 0], cn_vw[m + 1, 1]
            # print(f"m={m}, cm1={cm1}, cm2={cm2}, cmm1={cmm1}, cmm2={cmm2}")
            # early prob in late scenario
            joint_prob_log_el = np.logaddexp(joint_prob_log_el, log_xi_late_centroid[m, 2, cm2, cmm1, 2, cm1, cm2])
            # late prob in late scenario (centroid 1 -> 2, 2 -> 3)
            joint_prob_log_ll = np.logaddexp(joint_prob_log_ll,
                                             log_xi_late_centroid[m, cm1, cm1, cm2, cmm1, cmm1, cmm2])
            # late prob in early scenario
            joint_prob_log_le = np.logaddexp(joint_prob_log_le,
                                             log_xi_early_centroid[m, cm1, cm1, cm2, cmm1, cmm1, cmm2])
            # early prob in early scenario (centroid 2 -> 2)
            joint_prob_log_ee = np.logaddexp(joint_prob_log_ee, log_xi_early_centroid[m, 2, cm2, cmm1, 2, cm1, cm2])
            # self.assertEqual(np.argmax(log_xi_late_centroid[m, cm1, cm1, cm2, :, cmm1, cmm2]), cmm1, f"m={m}")
            # print(f"log(xi_[{m}, :, {cm1}, {cm2}, :, {cmm1}, {cmm2}]) = "
            #       f"{log_xi_late_centroid[m, :, cm1, cm2, :, cmm1, cmm2]}")
            self.assertEqual(np.argmax(log_xi_late_centroid[m, :, cm1, cm2, cmm1, cmm1, cmm2]), cm1, f"m={m}")
            # self.assertGreater(log_xi_late_centroid[m, cm1, cm1, cm2, cmm1, cmm1, cmm2],
            #                    log_xi_early_centroid[m, 2, cm1, cm2, 2, cmm1, cmm2],
            #                    f"m={m}")
            # The above assert does not pass, still to be further investigated
            # NOTE: it should be more likely to have a late centroid in the late scenario than a
            #   early centroid in the early scenario
            # Although this doesn't happen in aggregated probabilities (see below)

        # higher prob for a centroid closer to the leaves than centroid closer to the root
        self.assertGreater(joint_prob_log_ll, joint_prob_log_el)
        # higher prob for long l -> short l with centroid cn = leaves cn, than short l -> long l and centroid = leaves
        self.assertGreater(joint_prob_log_ll, joint_prob_log_le)
        self.assertGreater(joint_prob_log_ll, joint_prob_log_ee)

    def test_compute_exp_changes(self):
        # random seed
        random.seed(101)
        # generate cn
        n_sites = 1000
        t0 = n_sites // 200  # first change idx
        n_states = 4
        eps_ru = 200 / n_sites
        eps_uv = 0.01 / n_sites
        eps_uw = 100 / n_sites
        l_ru = l_from_p(eps_ru, n_states)
        l_uv = l_from_p(eps_uv, n_states)  # close to zero since no change from u to v
        l_uw = l_from_p(eps_uw, n_states)
        print(f"1 - pdd_ru: {1 - p_delta_change(n_states, l_ru, change=False)} - l_ru: {l_ru}")
        print(f"1 - pdd_uv: {1 - p_delta_change(n_states, l_uv, change=False)} - l_uv: {l_uv}")
        print(f"1 - pdd_uw: {1 - p_delta_change(n_states, l_uw, change=False)} - l_uw: {l_uw}")
        alpha = 1.

        centroid_cn = ([2] * t0 + [1] * t0) * 100  # 2 * 100 changes from root
        v_cn = ([2] * t0 + [1] * t0) * 100  # no distance from centroid
        w_cn = ([2] * t0 + [3] * (t0 // 2) + [1] * (t0 // 2 + 1)) * 100  # 100 more changes than v

        v_obs = emit_raw_obs(v_cn)
        w_obs = emit_raw_obs(w_cn)

        print(f"V obs: {v_obs[:20]}")
        print(f"W obs: {w_obs[:20]}")

        # compute expected changes given true l
        d, dp = compute_exp_changes(np.array([l_ru, l_uv, l_uw]), np.stack([v_obs, w_obs], axis=1),
                                    n_states=n_states, alpha=alpha)
        print(f"expected p statistic: p: {d / (d + dp)}"
              f" D = {d}, D' = {dp}")

        # test with eps model
        d, dp = compute_exp_changes(np.array([eps_ru, eps_uv, eps_uw]), np.stack([v_obs, w_obs], axis=1),
                                    n_states=n_states, alpha=alpha, jcb=False)
        print(f"expected p statistic via eps ({[eps_ru, eps_uv, eps_uw]}:\n"
              f"\tp: {d / (d + dp)}, D = {d}, D' = {dp}")

    def test_two_cells_eps(self):
        # TODO: make two cells experiment with eps model
        pass
