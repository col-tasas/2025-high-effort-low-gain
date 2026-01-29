"""
Class of discrete-time LTI systems
"""

__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "25/08/15"

import numpy as np


class LTI_discrete:
    def __init__(self, A: np.ndarray, B: np.ndarray):
        """
        Init LTI instance
        Inputs:
            - A: A-matrix
            - B: B-matrix
        """

        self.A = A
        self.B = B
        self.n_x, _ = A.shape
        _, self.n_u = B.shape

    def sim(self, x: np.ndarray, u: np.ndarray, w: np.ndarray, tSim):
        x_p = np.zeros((self.n_x, tSim))
        for t in range(tSim):
            x_p[:, [t]] = self.A @ x + self.B @ u[:, [t]] + w[:, [t]]
            x = x_p[:, [t]]

        return x_p
