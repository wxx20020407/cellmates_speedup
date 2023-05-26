import numpy as np


class Quadruplet():
    """
    Class representing the quadruplet object #, with associated C^#,
    """

    def __init__(self, M, A, C_r, yv, yw):
        self.M = M
        self.A = A
        self.C_r = C_r
        self.yv = yv
        self.yw = yw
        self.C_u = np.zeros((M, A))
        self.C_v = np.zeros((M, A))
        self.C_w = np.zeros((M, A))


