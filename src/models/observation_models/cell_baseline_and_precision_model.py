import numpy as np


class QuadrupletSpecificCellbaselineAndPrecisionModel:
    """
    Implementation of the Cell baseline and precision model.
    p(y_m^v |C^v, mu_v, tau_v) = N(y_vm |mu_v * C_m^v, 1/tau_v)
    """

    def __init__(self, M, mu_v_prior, mu_w_prior, tau_v_prior, tau_w_prior):
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

    def simulate_data(self, c_v, c_w, mu_v=None, mu_w=None, tau_v=None, tau_w=None):
        """
        Simulate data for the model using prior parameters as default.
        """
        mu_v = mu_v if mu_v is not None else self.mu_v_prior
        mu_w = mu_w if mu_w is not None else self.mu_w_prior
        tau_v = tau_v if tau_v is not None else self.tau_v_prior
        tau_w = tau_w if tau_w is not None else self.tau_w_prior

        y_v = np.zeros(self.M)
        y_w = np.zeros(self.M)
        for m in range(self.M):
            y_v[m] = np.random.normal(mu_v * c_v[m], 1./tau_v**(1/2))
            y_w[m] = np.random.normal(mu_w * c_w[m], 1./tau_w**(1/2))

        self.true_mu_v = mu_v
        self.true_mu_w = mu_w
        self.true_tau_v = tau_v
        self.true_tau_w = tau_w
        self.y_v = y_v
        self.y_w = y_w
        return y_v, y_w, mu_v, mu_w, tau_v, tau_w
