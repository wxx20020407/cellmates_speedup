import argparse
import logging
import pickle

import anndata
import numpy as np
from dendropy.calculate import treecompare
import dendropy as dpy
from matplotlib import pyplot as plt

from cellmates.models.evo import SimulationEvoModel, JCBModel
from cellmates.models.obs import NormalModel
from cellmates.other_methods import dice_api, medicc2_api
from cellmates.simulation import datagen
from cellmates.utils import visual, tree_utils


def simulate_data(N, M, K, n_clonal_events_per_edge, n_focal_events_per_edge,
                  clonal_CN_length_ratio=0.1, haplotype_aware=True,
                  out_dir='./simulated_data'):
    root_CN = 1 if haplotype_aware else 2
    evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                       n_focal_events=n_focal_events_per_edge,
                                       clonal_CN_length_ratio=clonal_CN_length_ratio,
                                       root_cn=root_CN)
    out_sim_hap_a = datagen.rand_dataset(K, M, evo_model_sim, obs_model='normal', n_cells=N)
    true_tree = out_sim_hap_a['tree']
    if haplotype_aware:
        out_sim_hap_b = datagen.rand_dataset(K, M, evo_model_sim,
                                             obs_model='normal', n_cells=N, tree=true_tree)
        cnps_hap_a = out_sim_hap_a['cn']
        cnps_hap_b = out_sim_hap_b['cn']
        cnps = np.stack((cnps_hap_a, cnps_hap_b), axis=-1)  # shape (2*n_cells-1, n_bins, 2)
        x_hap_a = out_sim_hap_a['obs']
        x_hap_b = out_sim_hap_b['obs']
        x = np.stack((x_hap_a, x_hap_b), axis=-1)  # shape (n_cells, n_bins, 2)
    else:
        cnps = out_sim_hap_a['cn']  # shape (2*n_cells-1, n_bins)
        x = out_sim_hap_a['obs']    # shape (n_cells, n_bins)

    # Save simulated data for visualization
    fig, ax = plt.subplots(2, 1)
    visual.plot_cn_profile(cnps_hap_a, cell_labels=np.arange(0, N), ax=ax[0], title="Hap A")
    visual.plot_cn_profile(cnps_hap_b, cell_labels=np.arange(0, N), ax=ax[1], title="Hap B")
    fig.savefig(out_dir + '/cn_profile.png')

    # Save true tree figure and Newick
    visual.plot_tree_phylo(true_tree, out_dir=out_dir, filename='true_tree', show=False)
    true_tree_nwk_file_path = out_dir + '/true_tree.nwk'
    with open(true_tree_nwk_file_path, 'w') as f:
        f.write(true_tree.as_string(schema='newick'))

    return cnps, true_tree, x

def convert_data(dataset_path, out_dir=None):
    out_dir = out_dir if out_dir is not None else dataset_path
    dice_api.convert_to_dice_tsv(cnps_obs, chr_ends_idx, bin_length, dataset_path + '_states.tsv')
    dice_api.convert_dice_tsv_to_medicc2()

def run_dice(dataset_path):
    dice_api.run_dice(dice_input_path, dice_out_dir, method='star', tree_rec='balME')

def run_medicc2(dataset_path):
    medicc2_api.run_medicc2(medicc2_input_path, medicc2_out_dir)

def run_cellmates_ideal(cnps, haplotype_aware, true_tree):
    # Compare with ideal Cellmates inference
    # Setup Cellmates model
    true_tree_nx = tree_utils.convert_dendropy_to_networkx(true_tree)
    evo_model = JCBModel(n_states=K)
    obs_model = NormalModel(n_states=K)
    cnps_cellmates = np.concatenate((cnps_hap_a, cnps_hap_b), axis=1)  # shape (n_cells, 2*n_bins)
    cell_pairs = list(itertools.combinations(range(N), r=2))
    psi_init = {'mu_v': 1.0, 'tau_v': 50.0, 'mu_w': 1.0, 'tau_w': 50.0}
    results, D, Dp = testing.run_ideal_cellmates_em_from_cnps(cnps_cellmates,
                                                              cnps_cellmates,
                                                              true_tree_nx, cell_pairs, K,
                                                              evo_model, obs_model, psi_init)

    distances = -np.ones((N, N, 3))
    iterations = -np.ones((N, N))
    loglikelihoods = -np.ones((N, N))
    # collect results
    for (u, v), l_i, loglik, it, _, _ in results:
        distances[u, v, :] = l_i
        iterations[(u, v)] = it
        loglikelihoods[(u, v)] = loglik


def compare_trees(true_tree, dice_tree, medicc2_tree, cellmates_tree, out_dir):
    # Calculate RF distances
    norm_rf_dist_dice = tree_utils.normalized_rf_distance(true_tree, dice_tree)
    norm_rf_dist_medicc2 = tree_utils.normalized_rf_distance(true_tree, medicc2_tree)
    norm_rf_dist_CM = tree_utils.normalized_rf_distance(true_tree, cellmates_tree)
    rf_dist_DICE = treecompare.symmetric_difference(true_tree, dice_tree)
    rf_dist_Medicc2 = treecompare.symmetric_difference(true_tree, medicc2_tree)
    rf_dist_CM = treecompare.symmetric_difference(true_tree, cellmates_tree)
    logging.info(f"Normalized RF distances:\nDICE: {norm_rf_dist_dice}\nMEDICC2: {norm_rf_dist_medicc2}\nCellmates: {norm_rf_dist_CM}")
    logging.info(f"RF distances:\nDICE: {rf_dist_DICE}\nMEDICC2: {rf_dist_Medicc2}\nCellmates: {rf_dist_CM}")
    # Save RF distances to file
    rf_dist = {'DICE': {'normalized': norm_rf_dist_dice, 'absolute': rf_dist_DICE},
               'MEDICC2': {'normalized': norm_rf_dist_medicc2, 'absolute': rf_dist_Medicc2},
               'Cellmates': {'normalized': norm_rf_dist_CM, 'absolute': rf_dist_CM}}
    with open(out_dir + '/rf_distances.pkl', 'w') as f:
        pickle.dump(rf_dist, f)

    # Save tree figures and Newick files
    visual.plot_tree_phylo(dice_tree, out_dir=out_dir, filename='dice_tree', show=False)
    visual.plot_tree_phylo(medicc2_tree, out_dir=out_dir, filename='medicc2_tree', show=False)
    visual.plot_tree_phylo(cellmates_tree, out_dir=out_dir, filename='cellmates_tree', show=False)
    dice_tree_nwk_file_path = out_dir + '/dice_tree.nwk'
    medicc2_tree_nwk_file_path = out_dir + '/medicc2_tree.nwk'
    cellmates_tree_nwk_file_path = out_dir + '/cellmates_tree.nwk'
    with open(dice_tree_nwk_file_path, 'w') as f:
        f.write(dice_tree.as_string(schema='newick'))
    with open(medicc2_tree_nwk_file_path, 'w') as f:
        f.write(medicc2_tree.as_string(schema='newick'))
    with open(cellmates_tree_nwk_file_path, 'w') as f:
        f.write(cellmates_tree.as_string(schema='newick'))


def load_trees(out_dir):
    # Load DICE tree
    dice_api.load_dice_tree(out_dir)
    medicc2_api.load_medicc2_tree(out_dir)

def run(args):
    if args.dataset_path is None:
        N, M, K = args.num_cells, args.num_bins, args.num_states
        nCN, nfCN = 1, 1
        haplotype_aware = args.haplotype_aware
        cnps, true_tree = simulate_data(N, M, K, nCN, nfCN, out_dir=args.out_dir)
    else:
        # Load dataset from path
        dataset_path = args.dataset_path
        adata = anndata.load_h5ad(dataset_path)
    # Convert dataset to DICE and MEDICC2 formats
    convert_data(dataset_path, args.out_dir)

    # Run DICE
    run_dice(dataset_path)
    # Run MEDICC2
    #run_medicc2(dataset_path)
    # Run Cellmates
    run_cellmates_ideal(cnps, haplotype_aware, true_tree)

    # Evaluation
    load_trees(out_dir)
    compare_trees(true_tree, dice_tree, medicc2_tree, cellmates_tree)


if __name__ == '__main__':
    cli = argparse.ArgumentParser("Run Cellmates, DICE and Medicc2 on simulated data")

    cli.add_argument('--out_dir', type=str, required=True,)
    cli.add_argument('--dataset_path', type=str, default=None,)
    cli.add_argument('--num_cells', type=int, default=10,)
    cli.add_argument('--num_bins', type=int, default=1000,)
    cli.add_argument('--num_states', type=int, default=5,)
    cli.add_argument('--num_processors', type=int, default=1,)
    cli.add_argument('--haplotype_aware', action='store_true',)
    args = cli.parse_args()



