import unittest

from cellmates.models.evo import SimulationEvoModel
from cellmates.simulation.datagen import rand_dataset


class SimulationEvoModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_SimulationEvoModel(self):
        n_states = 7
        n_sites = 200
        n_cells = 2
        n_focal_events = 5
        n_clonal_events = 5
        evo_model_sim = SimulationEvoModel(n_clonal_CN_events=n_clonal_events, n_focal_events=n_focal_events)
        data = rand_dataset(n_states, n_sites, obs_model='normal', evo_model=evo_model_sim, n_cells=n_cells)
        print(data['cn'])