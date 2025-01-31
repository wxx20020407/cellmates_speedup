import itertools
import time
from time import sleep

import numpy as np
import multiprocessing as mp

from inference.em import EM
from models.evolutionary_models.jukes_cantor_breakpoint import JCBModel
from models.observation_models.read_counts_models import PoissonModel
from simulation.datagen import simulate_quadruplet, get_ctr_table
from utils.math_utils import p_from_l


def parse_p_change_list(p_change):
    if len(p_change) == 1:
        p_change_u = p_change_v = p_change_w = p_change[0]
    elif len(p_change) == 2:
        p_change_u = p_change[0]
        p_change_v = p_change[1]
        p_change_w = p_change[1]
    else:
        p_change_u = p_change[0]
        p_change_v = p_change[1]
        p_change_w = p_change[2]
    return p_change_u, p_change_v, p_change_w

def run_experiment(n_sites, gamma_params, seed, max_iter, file_name, n_states):
    i = seed
    print(f"n_sites: {n_sites}")
    print(f"gamma_params: {gamma_params}")
    p_change = []
    for a, b in gamma_params:
        mean_length = a * b
        p_change.append(p_from_l(mean_length))
        print(f"mean: {mean_length:.3f} (p = {p_from_l(mean_length):.3f}), var: {a * b ** 2:.3f}", end=" --- ")
    p_change_u, p_change_v, p_change_w = parse_p_change_list(p_change)
    print(f"p_change_u: {p_change_u:.5f}, p_change_v: {p_change_v:.5f}, p_change_w: {p_change_w:.5f}")
    evo_model = JCBModel(n_states=n_states)
    obs_model = PoissonModel(n_states=n_states)

    # generate data
    data = simulate_quadruplet(n_sites=n_sites, obs_model=obs_model, evo_model=evo_model,
                               gamma_params=gamma_params, seed=i)

    # run EM
    em = EM(n_states=n_states, obs_model=obs_model, evo_model=evo_model, tree_build='ctr')
    start_time = time.time()
    em.fit(data['obs'], max_iter=max_iter, num_processors=1)
    exec_time = time.time() - start_time


    gt_ctr_table = get_ctr_table(data['tree'])
    true_param_ll = em.compute_pair_likelihood(data['obs'], theta=gt_ctr_table[0, 1])

    ctr_table_delta = em.distances - gt_ctr_table
    print(
        f"[{i}] time: {exec_time:.2f}, average delta ctr_table_distance: {np.mean(np.abs(ctr_table_delta[0, 1])):.5f}")
    with open(file_name, 'a') as f:
        f.write(
            f"{i},{n_sites},{n_states},{p_change_u},{p_change_v},{p_change_w},{em.distances[0, 1, 0]},{em.distances[0, 1, 1]},{em.distances[0, 1, 2]},"
            f"{ctr_table_delta[0, 1, 0]},{ctr_table_delta[0, 1, 1]}," f"{ctr_table_delta[0, 1, 2]},{exec_time:.2f},{em._n_iterations[(0, 1)]},{em._loglikelihoods[(0, 1)]},{true_param_ll}\n")


def main():
    # parameters
    """ parameters for synth_performance.py
    max_iter = 40
    n_sites = 500
    n_states = 7
    alpha = 1.
    p_change_list = [0.001]
    n_cells_list = [100]
    n_datasets = 10
    """
    max_iter = 60
    parallel_experiments = 10 # cpus
    n_states = 7
    n_sites_list = [
        200,
        500,
        1000
    ]
    gamma_params_list = [
        # [(0.8, 0.01)],
        # [(1, 0.01)],
        # [(10, 0.0005)],
        # [(100, 0.00005)],
        [(1 * 200, 0.03 / 200)],
        [(1 * 500, 0.03 / 200)],
        [(1 * 800, 0.03 / 200)],
        [(1 * 1000, 0.03 / 200)],
        # [(1 * 1000, 0.01 / 500), (1 * 500, 0.03 / 200), (1 * 200, 0.008 / 100)]
    ]
    num_replicates = 5
    print(f"with n_states = {n_states}, alpha = 1.")
    # make filename unique to avoid overwriting
    file_name = f'quadruplet_accuracy_{time.strftime("%y%m%d%H%M%S")}.csv'
    with open(file_name, 'w') as f:
        f.write('seed,n_sites,n_states,p_change_u,p_change_v,p_change_w,lu_em,lv_em,lw_em,lu_err,lv_err,lw_err,exec_time,n_iter,loglik,true_ll\n')
    params = [(n_sites, gamma_params, seed, max_iter, file_name, n_states) for n_sites, gamma_params, seed in itertools.product(n_sites_list, gamma_params_list, range(num_replicates))]
    if parallel_experiments == 1:
        for param in params:
            run_experiment(*param)
    else:
        # parallelize with multiprocessing
        with mp.Pool(parallel_experiments) as pool:
            # main loop
            pool.starmap(run_experiment, params)


if __name__ == '__main__':
    main()