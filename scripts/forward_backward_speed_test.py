import time

import numpy as np
from cellmates.models.evo import JCBModel
from cellmates.models.obs import PoissonModel
from cellmates.simulation.datagen import simulate_quadruplet
from cellmates.utils.math_utils import l_from_p

def main():
    seed = 42
    n_sites = 200
    n_states = 6
    lengths = l_from_p(np.array([0.1, 0.2, 0.15]), n_states=n_states)
    jcb_model = JCBModel(n_states)
    jcb_model.theta = lengths
    obs_model = PoissonModel(n_states, 100)
    dat = simulate_quadruplet(n_sites, obs_model=obs_model, evo_model='jcb', seed=seed, n_states=n_states, edge_lengths=lengths)
    reps = 5

    # benchmark time for forward-backward and new (faster) implementation
    old_time = time.time()
    for _ in range(reps):
        log_xi, log_gamma = jcb_model.two_slice_marginals(dat['obs'], obs_model)
    old_likelihood = jcb_model.loglikelihood
    old_time = time.time() - old_time
    new_time = time.time()
    for _ in range(reps):
        log_xi, log_gamma = jcb_model.two_slice_marginals_fast(dat['obs'], obs_model, normalization=20)
    new_time = time.time() - new_time
    new_likelihood = jcb_model.loglikelihood
    print(f"Old time: {old_time/5:.4f} s per run")
    print(f"New time: {new_time/5:.4f} s per run")
    print(f"Speedup: {old_time/new_time:.2f}x")
    if not np.isclose(old_likelihood, new_likelihood):
        "ERROR: Likelihoods do not match!"


if __name__ == "__main__":
    main()