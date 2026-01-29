__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "25/08/15"

import numpy as np
from fcn.LTI import LTI_discrete


class Bandit:
    def __init__(self, systems: list[LTI_discrete]):
        """
        Init for bandit class.
        Input:
            - systems: list[LTI_discrete]: list of candidate systems
        """

        self.n_x = systems[0].n_x
        self.n_u = systems[0].n_u

        self.systems = systems
        self.nOpts = len(self.systems)

    def calcRes(self, x_p: np.ndarray, x: np.ndarray, u: np.ndarray, hor: int):
        """
        Inputs:
            - x_p: np.ndarray (n_x, horizon)
            - x: np.ndarray (n_x, horizon)
            - u: npndarray (n_u, horizon)
            - hor: int, horizon length
        Outputs:
            - res: np.ndarray (n_x, horizon, nsystems)
        """
        res = np.zeros((self.n_x, hor, self.nOpts))
        for ii, sys in enumerate(self.systems):
            for t in range(hor):
                res[:, [t], ii] = x_p[:, [t]] - sys.sim(
                    x=x[:, [t]], u=u[:, [t]], w=np.zeros((self.n_x, 1)), tSim=1
                )

        return res
