import unittest
import numpy as np

from cellmates.models.obs import NegBinomialModel


class TestNegBinomial(unittest.TestCase):

    def test_initialize_and_new(self):
        model = NegBinomialModel(n_states=5, mu_v_prior=2.0, r_v_prior=10.0)
        model.initialize()
        assert np.isclose(model.mu_v, 2.0)
        assert np.isclose(model.r_v, 10.0)
        new_model = model.new()
        assert isinstance(new_model, NegBinomialModel)
        assert new_model.n_states == model.n_states

    def test_log_emission_shapes(self):
        model = NegBinomialModel(n_states=4)
        model.initialize()
        obs = np.random.poisson(5, size=(10, 2))
        log_v, log_w = model.log_emission_split(obs)
        assert log_v.shape == (10, 4)
        assert log_w.shape == (10, 4)
        log_joint = model.log_emission(obs)
        assert log_joint.shape == (10, 4, 4)

    def test_sample_and_mstep_consistency(self):
        np.random.seed(0)
        model = NegBinomialModel(n_states=3, mu_v_prior=2.0, r_v_prior=5.0)
        cnp = np.random.randint(1, 4, size=(1, 50))
        samples = model.sample(cnp)
        assert samples.shape == (50, 1)
        # Mock responsibilities (uniform)
        gamma = np.ones((50, model.n_states))
        gamma /= gamma.sum(1, keepdims=True)
        obs_vw = np.column_stack([samples[:, 0], samples[:, 0]])  # duplicate channel
        conds = (gamma, gamma)
        out = model.M_step(obs_vw, conds)
        assert "mu_v" in out and "r_v" in out
        assert out["mu_v"] > 0 and out["r_v"] > 0

    def test_M_step_optim_converges(self):
        np.random.seed(1)
        model = NegBinomialModel(n_states=5, mu_v_prior=100, r_v_prior=8.0)
        y = np.random.negative_binomial(8, 0.8, size=100)
        gamma = np.ones((100, 2)) / 2
        obs_vw = np.column_stack([y, y])
        out = model.M_step(obs_vw, (gamma, gamma))
        assert np.isfinite(out["mu_v"])
        assert np.isfinite(out["r_v"])
        assert out["mu_v"] > 0
        assert out["r_v"] > 0

    def test_realistic_read_generation_and_mstep(self):
        """
        Simulate single-cell reads under realistic coverage:
        - ~1M reads per cell
        - 0.1x coverage
        - 200 kb bins (~15k bins for 3 Gb genome)
        - toy CN profile with simple gains/losses
        """
        np.random.seed(42)
        model = NegBinomialModel(n_states=5, mu_v_prior=20., r_v_prior=5.)
        model.initialize()

        # Genome size ~3e9 bp → bins = 3e9 / 200e3 = 15000
        n_bins = 15_000
        # Synthetic copy-number profile (1=loss, 2=diploid, 3=gain)
        cn_profile = np.ones(n_bins)
        cn_profile[3_000:6_000] = 3  # gain region
        cn_profile[9_000:12_000] = 1  # loss region
        cnp = cn_profile[None, :]  # shape (1, n_bins)

        # Generate reads per bin proportional to CN
        reads = model.sample(cnp)  # shape (n_bins, 1)
        total_reads = reads.sum()
        print(f"Total reads generated: {total_reads}")
        print("reads stats: mean {:.2f}, std {:.2f}".format(reads.mean(), reads.std()))

        # Observed counts for two leaves (duplicate same data for v,w)
        obs_vw = np.column_stack([reads[:, 0], reads[:, 0]])

        # generate posteriors from actual copy numbers
        gamma = np.zeros((n_bins, model.n_states))
        for state in range(model.n_states):
            gamma[:, state] = (cn_profile == (state + 1)).astype(float)
        # normalize
        gamma /= gamma.sum(1, keepdims=True)
        conds = (gamma, gamma)

        # initialize parameters away from true
        model.initialize({"mu_v": 3., "r_v": 10., "mu_w": 5., "r_w": 20.})

        # Run one realistic M-step
        # FIXME: M-step fails cause of minimize function ABNORMAL_TERMINATION_IN_LNSRCH
        for _ in range(5):
            out = model.M_step(obs_vw, conds)
            print(out)
        assert np.isfinite(out["mu_v"])
        assert np.isfinite(out["r_v"])
        assert out["mu_v"] > 0
        assert out["r_v"] > 0
