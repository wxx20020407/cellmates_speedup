import hmmlearn

class EM():

    def __init__(self, N, M, A, obs):
        self.N = N
        self.M = M
        self.A = A
        self.K = 2*N
        self.obs = obs

    def run(self):


