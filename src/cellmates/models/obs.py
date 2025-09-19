import itertools
import logging

import numpy as np
from scipy import stats as ss


class ObsModel:

    def __init__(self, n_states: int, **kwargs):
        self.n_states = n_states

    def sample(self, cnp, **kwargs):
        """
        Simulate data for the model using prior parameters as default, given the copy number states.

        Parameters
        ----------
        cn_vw, array of shape (n_sites, 2) with copy number states for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        array of shape (n_sites, 2) with observations for pair of leaves
        """
        return False

    def log_emission(self, obs_vw, **kwargs):
        """
        Compute the log probability of the observations for each site
        $$
        \log p(y_m^{vw} | C_m^v = i, C_m^w = j)
        $$

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        array of shape (n_sites, n_states, n_states), log p(y_m^{vw} | C_m^v = i, C_m^w = j) for each m, i, j
        """
        return False

    # @classmethod
    # def get_instance(cls, obs_model, n_states):
    #     """
    #     Factory method to get an instance of an observation model given a string.
    #     Parameters
    #     ----------
    #     obs_model
    #     n_states
    #
    #     Returns
    #     -------
    #
    #     """
    #     # avoid circular import (https://stackoverflow.com/questions/72569089/factory-composite-design-patterns-combo-in-python-circular-imports)
    #     from models.observation_models.normalized_read_counts_models import NormalModel
    #     from models.observation_models.read_counts_models import PoissonModel
    #
    #     if isinstance(obs_model, ObsModel):
    #         if obs_model.n_states != n_states:
    #             logging.warning(f"Number of states mismatch: {obs_model.n_states} != {n_states},"
    #                             f" keeping n_states = {obs_model.n_states} from the model object")
    #         return obs_model
    #     elif obs_model == 'normal':
    #         return NormalModel(n_states)
    #     elif obs_model == 'poisson':
    #         return PoissonModel(n_states)
    #     else:
    #         raise ValueError(f"Unknown observation model {obs_model}")
    #


class NormalModel(ObsModel):
    """
    Implementation of the Cell baseline and precision model.
    p(y_m^v |C^v, mu_v, tau_v) = N(y_vm |mu_v * C_m^v, 1/tau_v)
    """

    def __init__(self, n_states: int, mu_v_prior=1., mu_w_prior=None, tau_v_prior=50., tau_w_prior=None, M=200, **kwargs):
        self.M = M
        self.mu_v_prior = mu_v_prior
        self.mu_w_prior = mu_w_prior if mu_w_prior is not None else mu_v_prior
        self.tau_v_prior = tau_v_prior
        self.tau_w_prior = tau_w_prior if tau_w_prior is not None else tau_v_prior
        self.true_mu_v = None
        self.true_mu_w = None
        self.true_tau_v = None
        self.true_tau_w = None
        self.y_v = None
        self.y_w = None
        super().__init__(n_states, **kwargs)

    def simulate_pair_data(self, c_v, c_w, obs_param_v=None, obs_param_w=None):
        """
        Simulate data for two cells with the model using prior parameters as default.
        """
        mu_v = obs_param_v['mu'] if obs_param_v is not None else self.mu_v_prior
        mu_w = obs_param_w['mu'] if obs_param_w is not None else self.mu_w_prior
        tau_v = obs_param_v['tau'] if obs_param_v is not None else self.tau_v_prior
        tau_w = obs_param_w['tau'] if obs_param_w is not None else self.tau_w_prior

        y_vw = self.sample(np.array([c_v, c_w]), mu_tau_params=np.array([[mu_v, tau_v], [mu_w, tau_w]]))
        y_v = y_vw[0]
        y_w = y_vw[1]

        self.true_mu_v = mu_v
        self.true_mu_w = mu_w
        self.true_tau_v = tau_v
        self.true_tau_w = tau_w
        obs_param_v = {'mu': mu_v, 'tau': tau_v}
        obs_param_w = {'mu': mu_w, 'tau': tau_w}
        self.y_v = y_v
        self.y_w = y_w
        return y_v, y_w, obs_param_v, obs_param_w

    def sample(self, cnp: np.ndarray, mu_tau_params: np.ndarray = None, **kwargs):
        """
        Sample emissions from array of copy number profiles.
        Simulate data for the model using prior parameters as default.
        Parameters
        ----------
        cnp : np.ndarray, shape (n_cells, n_sites), copy numbers for each site m for batch of cells
        mu_tau_params: np.ndarray, shape (n_cells, 2) or (1, 2), mu and tau parameters for each cell or shared

        Returns
        -------
        np.ndarray, shape (n_sites, n_cells), read counts for each site m for batch of cells
        """
        n_cells = cnp.shape[0]
        n_sites = cnp.shape[1]

        if mu_tau_params is None:
            mu_tau_params = np.array([[self.mu_v_prior, self.tau_v_prior],
                                     [self.mu_w_prior, self.tau_w_prior]])
        elif mu_tau_params.shape[0] != cnp.shape[0] and mu_tau_params.shape[0] != 1:
            raise ValueError(f"mu_tau_params has shape {mu_tau_params.shape} but expected (1, 2) or (n_cells, 2)")
        mu = np.empty(n_cells)
        tau = np.empty(n_cells)
        if n_cells == 1:
            mu[:] = mu_tau_params[0, 0]
            tau[:] = mu_tau_params[0, 1]
        else:
            mu = mu_tau_params[:, 0]
            tau = mu_tau_params[:, 1]

        normalized_reads = np.random.normal(mu[:, None] * cnp, (tau**(-1/2))[:, None])
        # fix negative values
        normalized_reads = np.clip(normalized_reads, a_min=0, a_max=None)
        return normalized_reads.transpose()

    def log_emission_legacy(self, obs_vw, **kwargs):
        # Old implementation of log_emission method with for loops.
        normal_mean_eps = 1e-10  # not the epsilon parameter of main model
        n_sites = obs_vw.shape[0]
        log_emissions = np.empty((n_sites, self.n_states, self.n_states))
        lam = np.array([self.mu_v_prior, self.mu_w_prior])
        for m, i, j in itertools.product(range(n_sites), range(self.n_states), range(self.n_states)):
            # log p(y_m^v | . ) + log p(y_m^w | . )
            log_emissions[m, i, j] = ss.norm.logpdf(obs_vw[m], np.clip(lam * np.array([i, j]),
                                                                          a_min=normal_mean_eps, a_max=None)).sum()

        return log_emissions

    def log_emission(self, obs_vw, **kwargs):

        """
        Compute the log probability of the observations for each site
        $$
        \log p(y_m^{vw} | C_m^v = i, C_m^w = j)
        $$

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        array of shape (n_sites, n_states, n_states), log p(y_m^{vw} | C_m^v = i, C_m^w = j) for each m, i, j
        """
        n_sites = obs_vw.shape[0]
        log_emissions = np.empty((n_sites, self.n_states, self.n_states))
        mu = np.array([self.mu_v_prior, self.mu_w_prior])
        tau = np.array([self.tau_v_prior, self.tau_w_prior])
        cn_mesh = np.meshgrid(*[range(self.n_states)] * 2, indexing='ij')
        loc_param = mu[:, None, None] * cn_mesh
        # log p(y_m^v | . ) + log p(y_m^w | . )
        # loc param: (n_bins, n_states, n_states)
        log_emissions[...] = ss.norm.logpdf(obs_vw[..., None, None], loc=loc_param[None, ...], scale=(tau**(-1/2))[None, :, None, None]).sum(axis=1)

        return log_emissions


class PoissonModel(ObsModel):
    """
    Implementation of the quadruplet specifc Poisson emission model.
    p(r_m^v |C^v, lambda_v) = Poisson(r_vm | lambda_v)
    """

    def __init__(self, n_states: int, lambda_v_prior: float = 100., lambda_w_prior=None, **kwargs):
        """
        Initialize the model with Poisson parameters.
        Parameters
        ----------
        lambda_v_prior : float, Poisson parameter for the read counts r_v
        lambda_w_prior : float, Poisson parameter for the read counts r_w (default: lambda_v_prior)
        """
        self.lambda_v_prior = lambda_v_prior
        self.lambda_w_prior = lambda_w_prior if lambda_w_prior is not None else lambda_v_prior
        self.true_lambda_v = None
        self.true_lambda_w = None
        self.r_v = None
        self.r_w = None
        self.M = None
        super().__init__(n_states, **kwargs)

    def sample(self, cnp: np.ndarray, lambda_: np.ndarray | float = None, **kwargs):
        """
        Sample emissions from array of copy number profiles.
        Simulate data for the model using prior parameters as default.
        Parameters
        ----------
        cnp : np.ndarray, shape (n_cells, n_sites), copy numbers for each site m for batch of cells
        lambda_: np.ndarray or float or None, Poisson baseline parameter(s) for the read-counts,
         if None use model default

        Returns
        -------
        np.ndarray, shape (n_sites, n_cells), read counts for each site m for parent v and child w
        """
        if lambda_ is None:
            lambda_ = self.lambda_v_prior
        elif isinstance(lambda_, np.ndarray) and (lambda_.shape[0] != cnp.shape[0] and lambda_.shape[0] != 1):
            raise ValueError(f"lambda_ has shape {lambda_.shape} but expected (1,) or (n_cells,)")
        r_vw = np.random.poisson(lambda_ * cnp)
        return r_vw.transpose()

    def log_emission(self, obs_vw, **kwargs) -> np.ndarray:
        """
        Compute the log probability of the observations for each site
        $$
        \log p(y_m^{vw} | C_m^v = i, C_m^w = j)
        $$

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        array of shape (n_sites, n_states, n_states), log p(y_m^{vw} | C_m^v = i, C_m^w = j) for each m, i, j
        """
        pois_mean_eps = 1e-10
        n_sites = obs_vw.shape[0]
        log_emissions = np.empty((n_sites, self.n_states, self.n_states))
        lam = np.array([self.lambda_v_prior, self.lambda_w_prior])
        cn_mesh = np.meshgrid(*[range(self.n_states)] * 2, indexing='ij')
        poisson_params = np.clip(lam[:, None, None] * cn_mesh, a_min=pois_mean_eps, a_max=None)
        # log p(y_m^v | . ) + log p(y_m^w | . )
        # loc param: (n_bins, n_states, n_states)
        log_emissions[...] = ss.poisson.logpmf(obs_vw[..., None, None], poisson_params[None, ...]).sum(axis=1)

        return log_emissions

class UrnModel(ObsModel):

    def __init__(self, n_states: int, **kwargs):
        raise NotImplementedError("UrnModel is not implemented yet")
        super().__init__(n_states, **kwargs)

    def sample(self, cnp, **kwargs):
        return super().sample(cnp, **kwargs)

    def log_emission(self, obs_vw, **kwargs):
        return super().log_emission(obs_vw, **kwargs)

    def simulate_data(self, R_0, min_gc, max_gc, c):
        R = self.simulate_total_reads(R_0)
        gc = self.simulate_gc_site_corrections(min_gc, max_gc)
        x, phi = self.simulate_observations_Poisson(c, R, gc)
        return x, R, gc, phi


    def simulate_total_reads(self, R_0) -> np.array:
        a = int(R_0 - R_0 / 10.)
        b = int(R_0 + R_0 / 10.)
        logging.debug(f"Reads per cell simulation: R in  [{a},{b}] ")
        R = ss.randint(a, b).rvs(self.N)
        return R


    def simulate_gc_site_corrections(self, min_gc=0.8, max_gc=1.2):
        logging.debug(f"GC correction per site simulation: g_m in  [{min_gc},{max_gc}] ")
        gc_dist = ss.uniform(min_gc, max_gc)
        gc = gc_dist.rvs((self.M,))
        return gc


    def simulate_observations_Poisson(self, c, R, gc):
        x = np.zeros((self.N, self.M))
        phi = np.einsum("nm, m -> n", c, gc)
        for n in range(self.N):
            mu_n = c[n] * gc * R[n] / phi[n]
            x_n_dist = ss.poisson(mu_n)
            x[n] = x_n_dist.rvs()

        return x, phi
