import logging


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
