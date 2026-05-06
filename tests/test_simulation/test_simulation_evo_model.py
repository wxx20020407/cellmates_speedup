import unittest

from matplotlib import pyplot as plt

from cellmates.models.evo import SimulationEvoModel
from cellmates.simulation.datagen import rand_dataset
from cellmates.utils import math_utils, tree_utils, testing, visual


class SimulationEvoModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_SimulationEvoModel_fixed_number_of_events(self):
        n_states = 7
        n_sites = 2000
        n_cells = 4
        n_focal_events = 5
        n_clonal_events = 5
        clonal_CN_length = 100
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events,
                                           n_focal_events=n_focal_events,
                                           clonal_CN_length=clonal_CN_length)
        data = rand_dataset(n_states, n_sites, obs_model='normal', evo_model=evo_model_sim, n_cells=n_cells)
        cnps = data['cn']
        tree = data['tree']
        tree_nx = tree_utils.convert_dendropy_to_networkx(tree)
        # check that the number of events is correct
        cn_changes_list = math_utils.compute_cn_changes(cnps, list(tree_nx.edges()))
        print(cn_changes_list)

        for u,v in tree_nx.edges():
            changes_start_uv = evo_model_sim.clonal_CN_events_start_pos[u, v]
            changes_end_uv = evo_model_sim.clonal_CN_events_end_pos[u, v]
            self.assertEqual(changes_start_uv.shape, changes_end_uv.shape)
            self.assertEqual(changes_start_uv.shape, (n_clonal_events,))

        # Plot and save cnps
        out_dir = testing.create_output_test_folder(sub_folder_name=f"M_{n_sites}")
        fig, ax = plt.subplots()
        visual.plot_cn_profile(cnps, ax=ax)
        fig.savefig(out_dir + '/cn_profile.png')

    def test_SimulationEvoModel_random_number_of_events(self):
        pass