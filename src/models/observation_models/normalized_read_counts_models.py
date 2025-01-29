import itertools

import numpy as np
import scipy.stats as ss

from models.observation_models import ObsModel


class NormalModel(ObsModel):
    """
    Implementation of the Cell baseline and precision model.
    p(y_m^v |C^v, mu_v, tau_v) = N(y_vm |mu_v * C_m^v, 1/tau_v)
    """

    def __init__(self, n_states: int, mu_v_prior=1., mu_w_prior=1., tau_v_prior=1., tau_w_prior=1., M=200, **kwargs):
        self.M = M
        self.mu_v_prior = mu_v_prior
        self.mu_w_prior = mu_w_prior
        self.tau_v_prior = tau_v_prior
        self.tau_w_prior = tau_w_prior
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

        if mu_tau_params is None:
            mu_tau_params = np.array([[self.mu_v_prior, self.tau_v_prior],
                                     [self.mu_w_prior, self.tau_w_prior]])
        elif mu_tau_params.shape[0] != cnp.shape[0] and mu_tau_params.shape[0] != 1:
            raise ValueError(f"mu_tau_params has shape {mu_tau_params.shape} but expected (1, 2) or (n_cells, 2)")
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
