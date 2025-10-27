import time
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

import numpy as np
import seaborn as sns
from cellmates.models.evo import JCBModel
from cellmates.models.obs import PoissonModel, NormalModel
from cellmates.simulation.datagen import simulate_quadruplet
from cellmates.utils.math_utils import l_from_p

def make_obs_models(n_states, mu, tau):
    obs_models = []
    for ijk in range(n_states**3):
        i = ijk // (n_states**2)
        j = (ijk // n_states) % n_states
        k = ijk % n_states
        mean_j = mu * j
        mean_k = mu * k
        # observation is a pair y_j, y_k ~ N(mu*j, tau) , N(mu*k, tau) independently
        obs_models.append(Normal((mean_j, mean_k), (tau, tau), covariance_type='diag'))
    return obs_models

def make_emissions(X, n_states, mu, tau):
    n_samples = 1
    seq_length = X.shape[0]
    emissions = np.zeros((n_samples, seq_length, n_states**3))
    # for each m and each possible state ijk, compute the log prob of observing X[m] given state ijk
    for ijk in range(n_states**3):
        i = ijk // (n_states**2)
        j = (ijk // n_states) % n_states
        k = ijk % n_states
        mean_j = mu * j
        mean_k = mu * k
        # observation is a pair y_j, y_k ~ N(mu*j, tau) , N(mu*k, tau) independently
        obs_model = Normal((mean_j, mean_k), (tau, tau), covariance_type='diag')
        emissions[0, :, ijk] = obs_model.log_probability(X)
    return emissions


def main():
    seed = 42
    # n_sites = 200
    # n_states = 6
    reps = 5

    # gather times for different n_states and n_sites
    n_states_list = [4, 6, 8, 10]
    n_sites_list = [100, 200, 500, 1000]
    # n_states_list = [4]
    # n_sites_list = [100]
    out_file = "forward_backward_speed_test_results.csv"
    with open(out_file, "w") as f:
        f.write("n_states,n_sites,old_time_per_run,new_time_per_run,speedup,lik_diff\n")
    for n_states in n_states_list:
        lengths = l_from_p(np.array([0.1, 0.2, 0.15]), n_states=n_states)
        jcb_model = JCBModel(n_states)
        jcb_model.theta = lengths
        # obs_model = PoissonModel(n_states, 100)
        obs_model = NormalModel(n_states, mu=1., tau=20)
        for n_sites in n_sites_list:
            dat = simulate_quadruplet(n_sites, obs_model=obs_model, evo_model='jcb', seed=seed, n_states=n_states,
                                      edge_lengths=lengths)
            print(f"n_states: {n_states}, n_sites: {n_sites}")
            old_times = []
            old_likelihoods = []
            # benchmark time for forward-backward and new (faster) implementation
            for _ in range(reps):
                old_time = time.time()
                log_xi, log_gamma = jcb_model.two_slice_marginals(dat['obs'], obs_model)
                old_time = time.time() - old_time
                old_likelihoods.append(jcb_model.loglikelihood)
                old_times.append(old_time)
            new_time = time.time()
            # for _ in range(reps):
            #     log_xi, log_gamma = jcb_model.two_slice_marginals_fast(dat['obs'], obs_model, normalization=20)
            # new_time = time.time() - new_time
            # new_likelihood = jcb_model.loglikelihood
            # try pomegranate
            # convert from 6D to 2D
            trans_mat_2D = jcb_model.trans_mat.reshape((n_states**3, n_states**3))
            start_prob_1D = jcb_model.start_prob.flatten()
            obs_models = make_obs_models(n_states, mu=1., tau=20)
            emissions = make_emissions(dat['obs'], n_states, mu=1., tau=20)
            model = DenseHMM(edges=trans_mat_2D, starts=start_prob_1D, distributions=obs_models, ends=np.ones(n_states**3) / (n_states**3))
            new_times = []
            new_likelihoods = []
            for _ in range(reps):
                new_time = time.time()
                _, _, _, _, new_likelihood = model.forward_backward(dat['obs'], emissions=emissions)
                new_time = time.time() - new_time
                new_likelihoods.append(new_likelihood)
                new_times.append(new_time)
            print(f"Old time per run (mean +- std): {np.mean(old_times):.4f} +/- {np.std(old_times):.4f} s")
            print(f"New time per run (mean +- std): {np.mean(new_times):.4f} +/- {np.std(new_times):.4f} s")

            if any([not np.isclose(ol, nl) for ol, nl in zip(old_likelihoods, new_likelihoods)]):
                "ERROR: Likelihoods do not match!"
            for i in range(len(old_times)):
                with open(out_file, "a") as f:
                    speedup = old_times[i] / new_times[i]
                    lik_diff = old_likelihoods[i] - new_likelihoods[i]
                    f.write(f"{n_states},{n_sites},{old_times[i]:.6f},{new_times[i]:.6f},{speedup:.4f},{lik_diff.item():.6f}\n")

            print("Results written to", out_file)
            # plot results to check O(K^6 M) and check constants
            res_df = sns.load_dataset("csv", data=out_file)
            # plot over n_sites for each n_states (hue=n_states)
            sns.lineplot(data=res_df, x="n_sites", y="old_time_per_run", hue="n_states", marker="o", label="Old")
            sns.lineplot(data=res_df, x="n_sites", y="new_time_per_run", hue="n_states", marker="o", linestyle="--", label="New")
            sns.scatterplot(data=res_df, x="n_sites", y="old_time_per_run", hue="n_states", marker="o")
            sns.scatterplot(data=res_df, x="n_sites", y="new_time_per_run", hue="n_states", marker="o", linestyle="--")
            sns.plt.xlabel("Number of Sites")
            sns.plt.ylabel("Time per Run (s)")
            sns.plt.title("Forward-Backward Time Comparison")
            sns.plt.legend()
            sns.plt.savefig("forward_backward_speed_comparison.png")
            # plot over n_states for each n_sites (hue=n_sites)
            sns.lineplot(data=res_df, x="n_states", y="old_time_per_run", hue="n_sites", marker="o", label="Old")
            sns.lineplot(data=res_df, x="n_states", y="new_time_per_run", hue="n_sites", marker="o", linestyle="--", label="New")
            sns.scatterplot(data=res_df, x="n_states", y="old_time_per_run", hue="n_sites", marker="o")
            sns.scatterplot(data=res_df, x="n_states", y="new_time_per_run", hue="n_sites", marker="o", linestyle="--")
            sns.plt.xlabel("Number of States")
            sns.plt.ylabel("Time per Run (s)")
            sns.plt.title("Forward-Backward Time Comparison")
            sns.plt.legend()
            sns.plt.savefig("forward_backward_speed_comparison_states.png")



if __name__ == "__main__":
    main()