import itertools

import numpy as np
import scipy.stats as ss

from models.observation_models import ObsModel


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

    def simulate_data(self, c_v, c_w, lambda_v=None, lambda_w=None):
        """
        Simulate data for the model using prior parameters as default.
        """
        lambda_v = lambda_v if lambda_v is not None else self.lambda_v_prior
        lambda_w = lambda_w if lambda_w is not None else self.lambda_w_prior

        self.M = len(c_v)

        r_v = np.zeros(self.M)
        r_w = np.zeros(self.M)
        for m in range(self.M):
            r_v[m] = np.random.poisson(lambda_v * c_v[m])
            r_w[m] = np.random.poisson(lambda_w * c_w[m])

        self.true_lambda_v = lambda_v
        self.true_lambda_w = lambda_w
        self.r_v = r_v
        self.r_w = r_w
        return r_v, r_w, lambda_v, lambda_w

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
        pois_mean_eps = 1e-10
        n_sites = obs_vw.shape[0]
        log_emissions = np.empty((n_sites, self.n_states, self.n_states))
        lam = np.array([self.lambda_v_prior, self.lambda_w_prior])
        # TODO: vectorize this
        for m, i, j in itertools.product(range(n_sites), range(self.n_states), range(self.n_states)):
            # log p(y_m^v | . ) + log p(y_m^w | . )
            log_emissions[m, i, j] = ss.poisson.logpmf(obs_vw[m], np.clip(lam * np.array([i, j]),
                                                                          a_min=pois_mean_eps, a_max=None)).sum()

        return log_emissions

