import numpy as np

from models.observation_models import QuadObs


class QuadrupletSpecificPoissonModel(QuadObs):
    """
    Implementation of the quadruplet specifc Poisson emission model.
    p(r_m^v |C^v, lambda_v) = Poisson(r_vm | lambda_v)
    """

    def __init__(self, lambda_v_prior, lambda_w_prior=None):
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
