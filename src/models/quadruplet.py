import networkx as nx
import numpy as np

from models.copy_tree import CopyTree
from models.observation_models.normalized_read_counts_models import QuadrupletSpecificCellbaselineAndPrecisionModel


class Quadruplet():
    """
    Class representing the quadruplet object #, with associated C^#=(C^r, C^u, C^v, C^w) , y=(yv, yw), CN model and 
    emission model.
    """

    def __init__(self, M, A, C_r, yv, yw, CN_model=None, obs_model=None):
        self.quadruplet_graph = nx.DiGraph([(0, 1), (1, 2), (1, 3)])
        self.M = M
        self.A = A
        self.C_r = C_r
        self.yv = yv
        self.yw = yw
        self.C_u = np.zeros((M, A))
        self.C_v = np.zeros((M, A))
        self.C_w = np.zeros((M, A))
        self.CN_model = CN_model if CN_model is not None else CopyTree(M, A, self.quadruplet_graph)
        self.observation_model = obs_model if obs_model is not None else QuadrupletSpecificCellbaselineAndPrecisionModel(
            M, 1., 1., 10., 10.)

    def simulate_data(self, eps_a, eps_b, eps_0, mu_v, mu_w, tau_v, tau_w):
        """
        Simulate data for the quadruplet using the CN model and the observation model.
        """
        eps, c = self.CN_model.simulate_data(eps_a=eps_a, eps_b=eps_b, eps_0=eps_0)
        c_v = c[2, :]
        c_w = c[3, :]
        yv, yw, mu_v, mu_w, tau_v, tau_w = self.observation_model.simulate_data(c_v, c_w, mu_v, mu_w, tau_v, tau_w)
        self.yv = yv
        self.yw = yw
        out = {'yv': yv, 'yw': yw, 'c_v': c_v, 'c_w': c_w, 'mu_v': mu_v, 'mu_w': mu_w, 'tau_v': tau_v, 'tau_w': tau_w}
        return out
