import unittest

import dendropy as dpy
from dendropy.calculate.treecompare import symmetric_difference
import networkx as nx

from cellmates.models.evo import SimulationEvoModel
from cellmates.simulation.datagen import rand_dataset
from cellmates.utils import tree_utils
import cellmates.utils.generalized_sankoff_algorithm as GS_alg


class GeneralizedSankoffAlgorithmTestCase(unittest.TestCase):

    def test_simulate_data_and_reconstruct_tree(self):
        # Simulation parameters
        n_sites = 1000
        n_cells = 20
        n_states = 7
        n_clonal_events_per_edge = 5
        n_focal_events_per_edge = 5
        clonal_CN_length = n_sites // 20
        obs_model_sim = 'normal'
        sim_evo_model = SimulationEvoModel(n_clonal_CN_events=n_clonal_events_per_edge,
                                           n_focal_events=n_focal_events_per_edge,
                                           clonal_CN_length=clonal_CN_length)
        data = rand_dataset(n_sites=n_sites, n_cells=n_cells, n_states=n_states,
                            obs_model=obs_model_sim,
                            evo_model=sim_evo_model)

        cnps = data['cn']
        leaf_cnps = cnps[:n_cells]
        tree_dp = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree_dp)

        # Prepare leaf CNPs dictionary
        cnps_internal_rec = GS_alg.reconstruct_cnps_with_block_mutations(tree_nx, leaf_cnps, max_mutations_per_edge=5)

        print("\n--- CNP Reconstruction Test ---")
        print(f"Difference per node between true and reconstructed CNPs:")
        for node in tree_nx.nodes():
            true_cnp = cnps_internal_rec[node]
            reconstructed_cnp = cnps_internal_rec[node]
            diff = (true_cnp != reconstructed_cnp).sum()
            print(f"Node {node}: {diff} sites differ")


if __name__ == '__main__':
    unittest.main()
