import itertools
import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as ss


class ObsModel(ABC):

    def __init__(self, n_states: int, train=False, **kwargs):
        self.n_states = n_states
        self.train = train
        self.psi = {}      # model parameters
        self.psi_init = {}  # initial parameters for EM
        self.psi_sim = {}  # parameters used for simulation

    def psi_array(self) -> np.ndarray:
        """
        Get model parameters as a flat array.
        """
        return np.array(list(self.psi.values()))

    @abstractmethod
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

    @abstractmethod
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
        return np.zeros((obs_vw.shape[0], self.n_states, self.n_states))

    @abstractmethod
    def log_emission_split(self, obs_vw, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the log probability of the observations for each site separately for v and w
        $$
        \log p(y_m^{v} | C_m^v = i)
        $$
        $$
        \log p(y_m^{w} | C_m^w = j)
        $$

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        tuple of two arrays each of shape (n_sites, n_states), log p(y_m^{v} | C_m^v = i) and log p(y_m^{w} | C_m^w = j) for each m, i and m, j
        """
        return (np.zeros((obs_vw.shape[0], self.n_states)),
                np.zeros((obs_vw.shape[0], self.n_states)))

    @abstractmethod
    def update(self, obs_vw, conditionals_vw, **kwargs):
        """
        Update model parameters after M-step.
        """
        pass

    @abstractmethod
    def M_step(self, obs_vw, conditionals_vw, **kwargs):
        """
        M-step to update model parameters given observations and posterior probabilities of copy number states.

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        posteriors_vw, array of shape (n_sites, n_states, n_states) with posterior probabilities of copy number states
        kwargs, additional parameters depending on the model

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def initialize(self, psi_init):
        pass

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

    def __init__(self, n_states: int,
                 mu_v_prior=1., mu_w_prior=None,
                 tau_v_prior=50., tau_w_prior=None,
                 M=200, train=False, **kwargs):
        super().__init__(n_states, train, **kwargs)
        self.M = M
        # Model Parameters
        self.mu_v = None
        self.mu_w = None
        self.tau_v = None
        self.tau_w = None
        # Priors for EM for Maximum a posteriori estimation
        self.mu_v_prior = mu_v_prior
        self.mu_w_prior = mu_w_prior if mu_w_prior is not None else mu_v_prior
        self.tau_v_prior = tau_v_prior
        self.tau_w_prior = tau_w_prior if tau_w_prior is not None else tau_v_prior
        self.update_params(self.mu_v_prior, self.tau_v_prior, self.mu_w_prior, self.tau_w_prior)

    def initialize(self, psi_init):
        if psi_init is not None:
            self.mu_v, self.tau_v = psi_init['mu_v'], psi_init['tau_v']
            self.mu_w, self.tau_w = psi_init['mu_w'], psi_init['tau_w']
        else:
            self.mu_v, self.tau_v = self.mu_v_prior, self.tau_v_prior
            self.mu_w, self.tau_w = self.mu_w_prior, self.tau_w_prior
        self.psi = {'mu_v': self.mu_v, 'tau_v': self.tau_v,
                           'mu_w': self.mu_w, 'tau_w': self.tau_w}
        self.psi_init = self.psi.copy()

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
        # use log_emission_split and combine
        log_emissions_v, log_emissions_w = self.log_emission_split(obs_vw, **kwargs)
        log_emissions[...] = log_emissions_v[:, :, None] + log_emissions_w[:, None, :]
        return log_emissions

    def log_emission_split(self, obs_vw, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the log probability of the observations for each site separately for v and w
        $$
        \log p(y_m^{v} | C_m^v = i)
        $$
        $$
        \log p(y_m^{w} | C_m^w = j)
        $$

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        tuple of two arrays each of shape (n_sites, n_states), log p(y_m^{v} | C_m^v = i) and log p(y_m^{w} | C_m^w = j) for each m, i and m, j
        """
        n_sites = obs_vw.shape[0]
        log_emissions_v = np.empty((n_sites, self.n_states))
        log_emissions_w = np.empty((n_sites, self.n_states))
        mu = np.array([self.mu_v, self.mu_w])
        tau = np.array([self.tau_v, self.tau_w])
        cn_states = np.arange(self.n_states)
        loc_param_v = mu[0] * cn_states
        loc_param_w = mu[1] * cn_states
        # log p(y_m^v | . )
        log_emissions_v[...] = ss.norm.logpdf(obs_vw[:, 0][:, None], loc=loc_param_v[None, :], scale=(tau[0]**(-1/2)))
        # log p(y_m^w | . )
        log_emissions_w[...] = ss.norm.logpdf(obs_vw[:, 1][:, None], loc=loc_param_w[None, :], scale=(tau[1]**(-1/2)))

        return log_emissions_v, log_emissions_w

    def update(self, obs_vw, conditionals_vw, **kwargs):
        """
        Update model parameters after M-step.
        """
        if not self.train:
            return
        out_M_step = self.M_step(obs_vw, conditionals_vw, **kwargs)
        mu_v, tau_v, mu_w, tau_w = out_M_step['mu_v'], out_M_step['tau_v'], out_M_step['mu_w'], out_M_step['tau_w']
        self.update_params(mu_v, tau_v, mu_w, tau_w)

    def M_step(self, obs_vw, conditionals_vw, **kwargs):
        """
        M-step to update model parameters given observations and conditional probabilities of the copy number states.
        Follows the eq:
        mu_v = (sum_m sum_i p(C_m^v = i | y_m^{vw}) * y_m^v) / (sum_m sum_i p(C_m^v = i | y_m^{vw}) * i)
        tau_v = 2*(sum_m sum_i p(C_m^v = i | y_m^{vw})) / (sum_m sum_i p(C_m^v = i | y_m^{vw}) * (y_m^v - mu_v * i)^2)
        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        posteriors_vw, array of shape (n_sites, n_states, n_states) with posterior probabilities of copy number states
        kwargs, additional parameters depending on the model
        Returns
        -------
        None
        """
        # update mu_v, tau_v
        y_v = obs_vw[:, 0]
        y_w = obs_vw[:, 1]
        gamma_v = conditionals_vw[0]
        gamma_w = conditionals_vw[1]
        cn_states = np.arange(self.n_states)
        mu_v = self.mu_update(gamma_v, y_v, cn_states)
        mu_w = self.mu_update(gamma_w, y_w, cn_states)
        tau_v = self.tau_update(gamma_v, y_v, cn_states, mu_v)
        tau_w = self.tau_update(gamma_w, y_w, cn_states, mu_w)
        return {'mu_v': mu_v, 'tau_v': tau_v, 'mu_w': mu_w, 'tau_w': tau_w}

    def mu_update(self, gamma, obs, cn_states):
        mu_num = np.einsum('mj, m, j ->', gamma, obs, cn_states)
        mu_den = np.einsum('mj, j ->', gamma, cn_states**2)
        mu = mu_num / mu_den
        return mu

    def tau_update(self, gamma, obs, cn_states, mu):
        obs_mu_c_diff = obs[:, None] - mu * cn_states[None, :]
        obs_mu_c_diff_squared = obs_mu_c_diff**2
        tau_num = 2 * np.einsum('mj ->', gamma)
        tau_den = np.einsum('mj, mj ->', gamma, obs_mu_c_diff_squared)
        tau = tau_num / tau_den
        return tau

    def update_params(self, mu_v, tau_v, mu_w, tau_w):
        self.mu_v = mu_v
        self.tau_v = tau_v
        self.mu_w = mu_w
        self.tau_w = tau_w
        self.psi = {'mu_v': self.mu_v, 'tau_v': self.tau_v,
                           'mu_w': self.mu_w, 'tau_w': self.tau_w}


class PoissonModel(ObsModel):
    """
    Implementation of the quadruplet specific Poisson emission model.
    p(r_m^v |C^v, lambda_v) = Poisson(r_vm | lambda_v)
    """

    def __init__(self, n_states: int,
                 lambda_v_prior: float = 100., lambda_w_prior=None,
                 train=False,
                 **kwargs):
        """
        Initialize the model with Poisson parameters.
        Parameters
        ----------
        lambda_v_prior : float, Poisson parameter for the read counts r_v
        lambda_w_prior : float, Poisson parameter for the read counts r_w (default: lambda_v_prior)
        """
        super().__init__(n_states, train, **kwargs)
        self.lambda_v = None
        self.lambda_w = None
        # Priors for EM for Maximum a posteriori estimation
        self.lambda_v_prior = lambda_v_prior
        self.lambda_w_prior = lambda_w_prior if lambda_w_prior is not None else lambda_v_prior
        self.true_lambda_v = None
        self.true_lambda_w = None
        self.M = None
        self.update_params(self.lambda_v_prior, self.lambda_w_prior)

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
        n_sites = obs_vw.shape[0]
        log_emissions = np.empty((n_sites, self.n_states, self.n_states))
        # use log_emission_split and combine
        log_emissions_v, log_emissions_w = self.log_emission_split(obs_vw, **kwargs)
        log_emissions[...] = log_emissions_v[:, :, None] + log_emissions_w[:, None, :]
        return log_emissions

    def log_emission_split(self, obs_vw, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the log probability of the observations for each site separately for v and w
        $$
        \log p(y_m^{v} | C_m^v = i)
        $$
        $$
        \log p(y_m^{w} | C_m^w = j)
        $$

        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        kwargs, additional parameters depending on the model

        Returns
        -------
        tuple of two arrays each of shape (n_sites, n_states), log p(y_m^{v} | C_m^v = i) and log p(y_m^{w} | C_m^w = j) for each m, i and m, j
        """
        pois_mean_eps = 1e-10
        n_sites = obs_vw.shape[0]
        log_emissions_v = np.empty((n_sites, self.n_states))
        log_emissions_w = np.empty((n_sites, self.n_states))
        lam = np.array([self.lambda_v, self.lambda_w])
        cn_states = np.arange(self.n_states)
        poisson_params_v = np.clip(lam[0] * cn_states, a_min=pois_mean_eps, a_max=None)
        poisson_params_w = np.clip(lam[1] * cn_states, a_min=pois_mean_eps, a_max=None)
        # log p(y_m^v | . )
        log_emissions_v[...] = ss.poisson.logpmf(obs_vw[:, 0][:, None], poisson_params_v[None, :])
        # log p(y_m^w | . )
        log_emissions_w[...] = ss.poisson.logpmf(obs_vw[:, 1][:, None], poisson_params_w[None, :])

        return log_emissions_v, log_emissions_w

    def update(self, obs_vw, conditionals_vw, **kwargs):
        """
        Runs the model M-step and updates the model parameters.
        """
        if not self.train:
            return
        out_M_step = self.M_step(obs_vw, conditionals_vw, **kwargs)
        lambda_v, lambda_w = out_M_step['lambda_v'], out_M_step['lambda_w']
        self.update_params(lambda_v, lambda_w)

    def M_step(self, obs_vw, conditionals_vw, **kwargs):
        """
        M-step to update model parameters given observations and conditional probabilities of the copy number states.
        Follows the eq:
        lambda_v = (sum_m sum_i p(C_m^v = i | y_m^{vw}) * y_m^v) / (sum_m sum_i p(C_m^v = i | y_m^{vw}) * i)
        Parameters
        ----------
        obs_vw, array of shape (n_sites, 2) with observations for pair of leaves
        posteriors_vw, array of shape (n_sites, n_states, n_states) with posterior probabilities of copy number states
        kwargs, additional parameters depending on the model
        Returns
        -------
        None
        """
        # update lambda_v, lambda_w
        y_v = obs_vw[:, 0]
        y_w = obs_vw[:, 1]
        gamma_v = conditionals_vw[0]
        gamma_w = conditionals_vw[1]
        cn_states = np.arange(self.n_states)
        lambda_v = self.lambda_update(gamma_v, y_v, cn_states)
        lambda_w = self.lambda_update(gamma_w, y_w, cn_states)
        return {'lambda_v': lambda_v, 'lambda_w': lambda_w}

    def lambda_update(self, gamma, obs, cn_states):
        lambda_num = np.einsum('mj, m ->', gamma, obs)
        lambda_den = np.einsum('mj, j ->', gamma, cn_states)
        lambda_ = lambda_num / lambda_den
        return lambda_

    def update_params(self, lambda_v, lambda_w):
        self.lambda_v = lambda_v
        self.lambda_w = lambda_w
        self.psi = {'lambda_v': self.lambda_v, 'lambda_w': self.lambda_w}

    def initialize(self, psi_init):
        if psi_init is not None:
            self.lambda_v = psi_init['lambda_v']
            self.lambda_w = psi_init['lambda_w']
        else:
            self.lambda_v = self.lambda_v_prior
            self.lambda_w = self.lambda_w_prior
        self.psi = {'lambda_v': self.lambda_v, 'lambda_w': self.lambda_w}


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
