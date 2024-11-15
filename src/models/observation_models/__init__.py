

class ObsModel:

    def __init__(self, n_states: int, **kwargs):
        self.n_states = n_states

    def simulate(self, cn_vw, **kwargs):
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

