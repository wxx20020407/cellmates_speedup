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
        cn_mesh = np.meshgrid(*[range(self.n_states)] * 2, indexing='ij')
        poisson_params = np.clip(lam[:, None, None] * cn_mesh, a_min=pois_mean_eps, a_max=None)
        # log p(y_m^v | . ) + log p(y_m^w | . )
        # loc param: (n_bins, n_states, n_states)
        log_emissions[...] = ss.poisson.logpmf(obs_vw[..., None, None], poisson_params[None, ...]).sum(axis=1)

        return log_emissions

