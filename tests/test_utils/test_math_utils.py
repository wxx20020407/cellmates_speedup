import unittest

import numpy as np

from cellmates.models.evo import SimulationEvoModel
from cellmates.models.obs import NormalModel
from cellmates.simulation import datagen
from cellmates.utils import tree_utils
from src.cellmates.utils import math_utils

class MathUtilsTestCase(unittest.TestCase):
    def test_flatten_unflatten_state(self):
        K = 5
        for l in range(K):
            for p in range(K):
                for o in range(K):
                    flat_index = math_utils._flatten_state(l, p, o, K)
                    l2, p2, o2 = math_utils._unflatten_state(flat_index, K)
                    self.assertEqual((l, p, o), (l2, p2, o2))

    def test_build_log_A_t(self):
        import numpy as np

        K = 3
        Delta_r_t = 1
        log_probs_ru = (np.log(0.9), np.log(0.1))
        log_probs_uv = (np.log(0.8), np.log(0.2))
        log_probs_uw = (np.log(0.7), np.log(0.3))

        log_A_t = math_utils._build_log_A_t(K, Delta_r_t, log_probs_ru, log_probs_uv, log_probs_uw)

        self.assertEqual(log_A_t.shape, (K, K, K, K, K, K))
        # Additional checks can be added here to verify specific values in log_A_t

    def test_viterbi_algorithm(self):
        # Additional checks can be added here to verify the correctness of the path
        # --- 1. Define Model Parameters ---
        M = 100  # Sequence length (Must be VERY small)
        K = 7  # Number of states per chain (Must be VERY small)

        # K=2 -> K^6 = 64
        # K=3 -> K^6 = 729
        # K=4 -> K^6 = 4,096
        # K=5 -> K^6 = 15,625
        # K=8 -> K^6 = 262,144 (This will require gigabytes of RAM per time step)

        print(f"Running Viterbi with M={M}, K={K}.")
        print(f"Total states N = K^3 = {K ** 3}")
        print(f"Memory for K^6 tensor (float64): ~{(K ** 6 * 8) / (1024 ** 2) :.2f} MB")

        # --- 2. Simulate Data ---
        obs_model = NormalModel(K, mu_v_prior=1.0, tau_v_prior=100.0)
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=5, n_focal_events=5, clonal_CN_length=M//20)
        data = datagen.simulate_quadruplet(M, obs_model, evo_model_sim, n_states=K)
        cnps = data['cn']
        obs = data['obs']


        # --- 2. Create Dummy Data ---
        Z_r = cnps[3, :]  # Use the first chain as Z^r
        log_emissions = obs_model.log_emission(obs)
        raw_pi = np.zeros((K, K, K))
        raw_pi[2, 2, 2] = 1.0  # Start in the CN=2 state
        log_pi = np.log(raw_pi / raw_pi.sum())

        eps = math_utils.get_expected_branch_lengths_from_cnps(cnps, K)
        eps_ru = eps[0,1]
        eps_uv = eps[1,2]
        eps_uw = eps[1,3]

        # --- 3. Run Viterbi Algorithm ---
        print("\nRunning (matrix) Viterbi...")
        best_path, max_log_prob = math_utils.viterbi_matrix_K6(
            log_emissions, Z_r, log_pi, eps_ru, eps_uv, eps_uw
        )

        print("Viterbi complete.")
        print(f"\nBest Path Log-Probability: {max_log_prob:.6f}")

        print("\nBest Path (l, p, o) for (Z^u, Z^v, Z^w):")
        for t in range(M):
            print(f"  t={t}: State {best_path[t]}")

        viterbi_path_u = best_path[:, 0]
        viterbi_path_v = best_path[:, 1]
        viterbi_path_w = best_path[:, 2]
        diff_u = np.abs(viterbi_path_u - cnps[2])
        diff_v = np.abs(viterbi_path_v - cnps[0])
        diff_w = np.abs(viterbi_path_w - cnps[1])
        print(f"\nDifference between Viterbi path and true CN (Z^u): {diff_u}")
        print(f"Difference between Viterbi path and true CN (Z^v): {diff_v}")
        print(f"Difference between Viterbi path and true CN (Z^w): {diff_w}")

    def test_viterbi_optimized_K5_algorithm(self):
        # Additional checks can be added here to verify the correctness of the path
        # --- 1. Define Model Parameters ---
        M = 100  # Sequence length (Must be VERY small)
        K = 7  # Number of states per chain (Must be VERY small)

        # K=2 -> K^6 = 64
        # K=3 -> K^6 = 729
        # K=4 -> K^6 = 4,096
        # K=5 -> K^6 = 15,625
        # K=8 -> K^6 = 262,144 (This will require gigabytes of RAM per time step)

        print(f"Running Viterbi with M={M}, K={K}.")
        print(f"Total states N = K^3 = {K ** 3}")
        print(f"Memory for K^6 tensor (float64): ~{(K ** 6 * 8) / (1024 ** 2) :.2f} MB")

        # --- 2. Simulate Data ---
        obs_model = NormalModel(K, mu_v_prior=1.0, tau_v_prior=100.0)
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=5, n_focal_events=5, clonal_CN_length=M // 20)
        data = datagen.simulate_quadruplet(M, obs_model, evo_model_sim, n_states=K)
        cnps = data['cn']
        obs = data['obs']

        # --- 2. Create Dummy Data ---
        Z_r = cnps[3, :]  # Use the first chain as Z^r
        log_emissions = obs_model.log_emission(obs)
        raw_pi = np.zeros((K, K, K))
        raw_pi[2, 2, 2] = 1.0  # Start in the CN=2 state
        log_pi = np.log(raw_pi / raw_pi.sum())

        eps = math_utils.get_expected_branch_lengths_from_cnps(cnps, K)
        eps_ru = eps[0, 1]
        eps_uv = eps[1, 2]
        eps_uw = eps[1, 3]

        # --- 3. Run Viterbi Algorithm ---
        print("\nRunning (matrix) Viterbi...")
        best_path, max_log_prob = math_utils.viterbi_optimized_K5(
            log_emissions, Z_r, log_pi, eps_ru, eps_uv, eps_uw
        )

        print("Viterbi complete.")
        print(f"\nBest Path Log-Probability: {max_log_prob:.6f}")

        print("\nBest Path (l, p, o) for (Z^u, Z^v, Z^w):")
        for t in range(M):
            print(f"  t={t}: State {best_path[t]}")

        viterbi_path_u = best_path[:, 0]
        viterbi_path_v = best_path[:, 1]
        viterbi_path_w = best_path[:, 2]
        diff_u = np.abs(viterbi_path_u - cnps[2])
        diff_v = np.abs(viterbi_path_v - cnps[0])
        diff_w = np.abs(viterbi_path_w - cnps[1])
        print(f"\nDifference between Viterbi path and true CN (Z^u): {diff_u}")
        print(f"Difference between Viterbi path and true CN (Z^v): {diff_v}")
        print(f"Difference between Viterbi path and true CN (Z^w): {diff_w}")

