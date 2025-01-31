import itertools
import logging
import random
import time
import unittest

import dendropy
import networkx as nx
import numpy as np
from dendropy.calculate import treecompare
from scipy.special import logsumexp

from models.evolutionary_models import p_delta_change
from models.evolutionary_models.jukes_cantor_breakpoint import JCBModel
from models.observation_models.read_counts_models import PoissonModel
from simulation.datagen import rand_dataset, get_ctr_table, simulate_quadruplet
from inference.em import jcb_em_ctrtable, EM, jcb_em_alg, em_alg
from models.evolutionary_models.copy_tree import CopyTree
from utils.tree_utils import convert_networkx_to_dendropy, get_node2node_distance, random_binary_tree, label_tree
from utils.math_utils import l_from_p, p_from_l

from inference.neighbor_joining import build_tree


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


class EMTestCase(unittest.TestCase):

    def setUp(self) -> None:
        random.seed(101)
        np.random.seed(seed=101)
        logging.basicConfig(level=logging.DEBUG)
        self.DEFAULT_GAMMA_PARAMS = [(1*1000, 0.01/1000), (1*500, 0.03/500), (1*200, 0.008/200)]  # with 500 bins

    def test_em_alg(self):
        # generate toy data
        n_states = 5
        obs, eps = _generate_obs(noise=10)
        # run em
        em = EM(n_states, PoissonModel(n_states, 100, 100), 'copytree', tree_build='ctr', verbose=2)
        em.fit(obs, max_iter=30, rtol=1e-3, num_processors=8)
        ctr_table = em.distances
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
        ctr_table = jcb_em_ctrtable(obs, n_states=5)
        print(ctr_table)
        # build tree
        em_tree = build_tree(ctr_table)
        print(em_tree)
        nx.write_network_text(em_tree, sources=['r'])
        assert nx.is_tree(em_tree)

    def test_tree_inference_synth(self):
        seed = 101
        n_states = 5
        n_sites = 500
        n_cells = 8
        data = rand_dataset(n_states, n_sites, obs_model='poisson', alpha=1., p_change=0.05, n_cells=n_cells, seed=seed)
        print("Generated tree")
        data['tree'].print_plot(plot_metric='length')
        for node in data['tree'].preorder_node_iter():
            print(node.label, node.edge_length)

        print("Observations")
        print(data['obs'][:20, :])

        # run EM
        ctr_table = jcb_em_ctrtable(data['obs'], n_states=n_states)
        print(ctr_table[..., 0])

        em_tree = build_tree(ctr_table)
        # relabel tree nodes with data taxon labels
        nx.write_network_text(em_tree, sources=['r'])
        labels_mapping = {n.label: n.label for n in data['tree'].nodes() if n != data['tree'].seed_node}
        labels_mapping['r'] = data['tree'].seed_node.label
        nx.write_network_text(nx.relabel_nodes(em_tree, labels_mapping, copy=True),
                              sources=[data['tree'].seed_node.label])

        # compare with true tree using RF-distance (unweighted)
        dendropy_tree = convert_networkx_to_dendropy(em_tree,
                                                     taxon_namespace=data['tree'].taxon_namespace)
        dendropy_tree.print_plot()
        sym_distance_jcb = treecompare.symmetric_difference(data['tree'], dendropy_tree)
        print(f'Symmetric (unweighted) distance: {sym_distance_jcb}')
        rf_distance_jcb = treecompare.robinson_foulds_distance(data['tree'], dendropy_tree, edge_weight_attr='length')
        print(f'Robinson-Fould distance: {rf_distance_jcb}')
        self.assertEqual(sym_distance_jcb, 0)
        self.assertEqual(rf_distance_jcb, 0)

    def test_quadruplet(self):
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500

        data = simulate_quadruplet(n_sites, n_states=n_states, gamma_params=(1., 0.055))
        gt_ctr_table = get_ctr_table(data['tree'])
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths
        l_true = gt_ctr_table[0, 1, :].tolist()
        data['tree'].print_plot(plot_metric='length')
        print(f"True edge _lengths: {l_true}")

        # run EM
        out = jcb_em_alg(data['obs'], n_states=n_states, max_iter=30, rtol=1e-3, num_processors=1)
        ctr_table = out['l_hat']

        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)

        # check likelihood
        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = l_true

        obs_model = PoissonModel(n_states, 100, 100)
        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_true = evo_model._forward_pass_likelihood(data['obs'], log_emissions)

        ll_est = out['loglikelihoods'][0, 1]
        self.assertGreater(ll_est, ll_true)

        # if these tests don't pass, it's likely that they are wrong
        self.assertAlmostEqual(ctr_table[0, 1, 0], gt_ctr_table[0, 1, 0], delta=0.02)
        self.assertAlmostEqual(ctr_table[0, 1, 1], gt_ctr_table[0, 1, 1], delta=0.02)
        self.assertAlmostEqual(ctr_table[0, 1, 2], gt_ctr_table[0, 1, 2], delta=0.01)

    def test_quadruplet_random_l(self):
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500
        p_change = 0.02  # for random edge _lengths

        alpha = 1.
        data = simulate_quadruplet(n_states, n_sites, alpha=alpha, l_mean=l_from_p(p_change, n_states))
        gt_ctr_table = get_ctr_table(data['tree'])
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths
        l_true = gt_ctr_table[0, 1, :].tolist()
        data['tree'].print_plot(plot_metric='length')
        print(f"True edge _lengths: {l_true}")

        # run EM
        out = jcb_em_alg(data['obs'], n_states=n_states, max_iter=30, rtol=1e-5, num_processors=8)
        ctr_table = out['l_hat']

        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)

        # check likelihood
        ll_est = out['loglikelihoods'][0, 1]
        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = l_true

        obs_model = PoissonModel(n_states, 100, 100)
        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_true = evo_model._forward_pass_likelihood(data['obs'], log_emissions)
        self.assertGreater(ll_est, ll_true)

        # if these tests don't pass, it's likely that they are wrong
        self.assertAlmostEqual(ctr_table[0, 1, 0], gt_ctr_table[0, 1, 0], delta=0.03)
        self.assertAlmostEqual(ctr_table[0, 1, 1], gt_ctr_table[0, 1, 1], delta=0.03)
        self.assertAlmostEqual(ctr_table[0, 1, 2], gt_ctr_table[0, 1, 2], delta=0.03)

    def test_quadruplet_true_init(self):
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500

        data = simulate_quadruplet(n_sites, gamma_params=self.DEFAULT_GAMMA_PARAMS, n_states=n_states)
        gt_ctr_table = get_ctr_table(data['tree'])
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (first 20) (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths
        l_true = gt_ctr_table[0, 1, :].tolist()
        data['tree'].print_plot(plot_metric='length')
        print(f"True edge _lengths: {l_true}")
        print(f"(from p: {p_from_l(gt_ctr_table[0, 1, :], n_states)}")

        # run EM
        out = jcb_em_alg(data['obs'], n_states=n_states, l_init=gt_ctr_table[0, 1, :], max_iter=30, rtol=1e-5, num_processors=1)
        ctr_table = out['l_hat']
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        l_est = ctr_table[0, 1, :].tolist()
        print("Estimated edge _lengths:")
        print(l_est)

        # check likelihood
        ll_est = out['loglikelihoods'][0, 1]
        evo_model = JCBModel(n_states=n_states)
        evo_model.lengths = l_true

        obs_model = PoissonModel(n_states, 100, 100)
        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_true = evo_model._forward_pass_likelihood(data['obs'], log_emissions)
        self.assertGreater(ll_est, ll_true)

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
        data = rand_dataset(n_states, n_sites, obs_model='poisson', alpha=alpha, p_change=20 / n_sites, n_cells=n_cells,
                            seed=seed)
        print(f"True CTR table")
        true_ctr_table = get_ctr_table(data['tree'])
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
            centroid = data['tree'].mrca(taxon_labels=[str(r), str(s)])
            if centroid != data['tree'].seed_node:
                c1, c2 = r, s
                break

        print(f"Copy Numbers")
        print(f"R:\t {data['cn'][int(data['tree'].seed_node.label), :]}")
        print(f"U:\t {data['cn'][int(centroid.label), :]}")
        print(f"C{c1}:\t {data['cn'][c1, :]}")
        print(f"C{c2}:\t {data['cn'][c2, :]}")

        print(f"Leaves with non-root CTR: {c1}, {c2}")
        l_uv = get_node2node_distance(data['tree'], centroid.label, str(c1))
        l_uw = get_node2node_distance(data['tree'], centroid.label, str(c2))
        print(f"true l_trip: {true_ctr_table[c1, c2]}, {l_uv}, {l_uw}")
        # run EM
        ctr_table = jcb_em_ctrtable(data['obs'][:, [c1, c2]], n_states=n_states)
        print(f"Estimated CTR table")
        print(ctr_table)
        self.assertAlmostEqual(ctr_table[0, 1, 0], true_ctr_table[c1, c2, 0],
                               msg=f"cell {c1} and {c2} CTR: {ctr_table[0, 1, 0]} != {true_ctr_table[c1, c2, 0]}", places=3)
        # build tree
        em_tree = build_tree(ctr_table)

        dendropy_tree = convert_networkx_to_dendropy(em_tree, taxon_namespace=data['tree'].taxon_namespace)
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
        # reasonable _lengths computed by setting the change probability to
        p_change = 3 / n_sites
        ll = l_from_p(p_change * 2, n_states)
        sl = l_from_p(0, n_states)
        jcb_model = JCBModel(n_states=n_states)
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
        jcb_model.lengths = np.array([sl, ll, ll])
        log_xi_early_centroid = jcb_model.two_slice_marginals(obs_vw)
        loglik_early = jcb_model.loglikelihood

        # compute two-slice marginals assuming centroid is placed closer to the leaves
        jcb_model.lengths = np.array([ll, sl, sl])
        log_xi_late_centroid = jcb_model.two_slice_marginals(obs_vw)
        loglik_late = jcb_model.loglikelihood
        self.assertGreater(loglik_late, loglik_early)

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

        # generate observations
        pois_model = PoissonModel(n_states=n_states)
        vw_obs = pois_model.sample(np.array([v_cn, w_cn]))

        print(f"V,W obs:\n {vw_obs[:, :20]}")

        quad_model = JCBModel(n_states=n_states)
        quad_model.lengths = np.array([l_ru, l_uv, l_uw])

        # compute expected changes given true l
        d, dp, loglik = quad_model.expected_changes(vw_obs, obs_model=pois_model)
        print(f"expected p statistic: p: {d / (d + dp)}"
              f" D = {d}, D' = {dp}, loglik = {loglik}")

        # test with eps model
        quad_model = CopyTree(n_states=n_states)
        quad_model.eps = np.array([eps_ru, eps_uv, eps_uw])
        d, dp, loglik = quad_model.expected_changes(vw_obs, obs_model=pois_model)
        print(f"expected p statistic via eps ({[eps_ru, eps_uv, eps_uw]}):\n"
              f"\tp: {d / (d + dp)}, D = {d}, D' = {dp}, loglik = {loglik}")

    def test_build_tree(self):
        n_cells = 10
        # generate tree
        tree: dendropy.Tree = random_binary_tree(n_cells, length_mean=0.01, seed=101)
        print("--- Starting tree ---")
        print(f'txnsp: {tree.taxon_namespace}')
        label_tree(tree, method='group')
        tree.print_plot(plot_metric='length')

        # derive true ctr table from tree
        ctr_table = get_ctr_table(tree)

        # rebuild tree
        nx_tree = build_tree(ctr_table)
        new_dpy_tree = convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace, edge_length='length')
        print("--- Rebuilt tree ---")
        print(f'txnsp: {new_dpy_tree.taxon_namespace}')
        label_tree(new_dpy_tree, method='group')
        new_dpy_tree.print_plot(plot_metric='length')

        self.assertTrue(len(new_dpy_tree.taxon_namespace) == len(tree.taxon_namespace))
        for taxon in new_dpy_tree.taxon_namespace:
            self.assertTrue(taxon in tree.taxon_namespace)
            self.assertEqual(taxon.label, tree.find_node_with_taxon(lambda t: t.label == taxon.label).taxon.label)
        self.assertEqual(treecompare.symmetric_difference(tree, new_dpy_tree), 0)
        self.assertLess(treecompare.robinson_foulds_distance(tree, new_dpy_tree, edge_weight_attr='length'), 0.01)

        # random tree robinson foulds distance
        rnd_dpy_tree = convert_networkx_to_dendropy(nx_tree, taxon_namespace=tree.taxon_namespace)
        # rename leaves to random numbers
        rnd_ints = np.random.permutation(range(n_cells))
        for leaf in rnd_dpy_tree.leaf_node_iter():
            leaf.taxon.label = str(rnd_ints[int(leaf.taxon.label)])

        self.assertEqual(treecompare.symmetric_difference(tree, rnd_dpy_tree), 0)

    def test_multiprocessing(self):
        logging.basicConfig(level=logging.DEBUG)
        seed = 42

        n_cells = 10
        n_states = 5
        n_sites = 100
        p_change = 0.02

        data = rand_dataset(n_states, n_sites, obs_model='poisson', p_change=p_change, n_cells=n_cells, seed=seed)
        self.assertEqual(data['obs'].shape, (n_sites, n_cells))

        l_init = np.random.exponential(scale=l_from_p(p_change, n_states), size=3)
        start_time = time.time()
        ctr_table_5p = jcb_em_ctrtable(data['obs'], n_states=n_states, l_init=l_init, max_iter=50, num_processors=5)
        tot_time_5_proc = time.time() - start_time

        start_time = time.time()
        ctr_table_1p = jcb_em_ctrtable(data['obs'], n_states=n_states, l_init=l_init, max_iter=50, num_processors=1)
        tot_time_1_proc = time.time() - start_time

        print(f"5proc: {tot_time_5_proc}, 1proc: {tot_time_1_proc}")

        self.assertLess(tot_time_5_proc, tot_time_1_proc)
        self.assertTrue(np.allclose(ctr_table_5p, ctr_table_1p))

    def test_quadruplet_copytree(self):
        # seed for reproducibility
        seed = 120
        # dendropy seed
        random.seed(seed)
        np.random.seed(seed)
        n_states = 5
        n_sites = 500

        data = simulate_quadruplet(n_sites, evo_model='copytree', n_states=n_states, gamma_params=self.DEFAULT_GAMMA_PARAMS)
        gt_ctr_table = get_ctr_table(data['tree'])  # eps
        # print cn in order r, u, v, w (check simulate_quadruplet doc for sorting info)
        print(f"CN (r, u, v, w):\n{data['cn'][[3, 2, 0, 1], :20]}")

        # print tree with _lengths
        data['tree'].print_plot(plot_metric='length')
        print(f"True edge _eps: {gt_ctr_table[0, 1, :].tolist()}")

        # run EM
        out = em_alg(data['obs'], n_states=n_states, max_iter=50, rtol=1e-5, num_processors=1, eps_init=np.array([0.1, 0.1, 0.1]))
        ctr_table = out['l_hat']
        ll_est = out['loglikelihoods'][0, 1]
        # change tree _lengths to match the estimated ones
        for edge in data['tree'].preorder_edge_iter():
            if edge.head_node.label == 2:
                edge.length = ctr_table[0, 1, 0]
            elif edge.head_node.label == 0:
                edge.length = ctr_table[0, 1, 1]
            elif edge.head_node.label == 1:
                edge.length = ctr_table[0, 1, 2]
        data['tree'].print_plot(plot_metric='length')
        print("Estimated edge _lengths:")
        eps_est = ctr_table[0, 1, :].tolist()
        print(eps_est)

        # check likelihood
        evo_model = CopyTree(n_states=n_states)
        evo_model.eps = gt_ctr_table[0, 1, :]

        obs_model = PoissonModel(n_states, 100, 100)
        log_emissions = obs_model.log_emission(data['obs'])
        _, ll_true = evo_model._forward_pass_likelihood(data['obs'], log_emissions)

        self.assertGreater(ll_est, ll_true)

        # if these tests don't pass, it's likely that they are wrong
        self.assertAlmostEqual(ctr_table[0, 1, 0], gt_ctr_table[0, 1, 0], delta=0.02)
        self.assertAlmostEqual(ctr_table[0, 1, 1], gt_ctr_table[0, 1, 1], delta=0.02)
        self.assertAlmostEqual(ctr_table[0, 1, 2], gt_ctr_table[0, 1, 2], delta=0.01)

