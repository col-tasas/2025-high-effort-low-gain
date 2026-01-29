"""
Class Definition of different Input design algorithms.

Instances of this class can
  - Generate random inputs
  - Yield input sequences as defined in Alg. 2
  - Yield input sequences according to the design criterion in [27]

"""

__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "25/08/15"


import numpy as np
from fcn.bandit import Bandit
from scipy.optimize import minimize

from fcn.LTI import LTI_discrete


class Explorer:
    def __init__(
        self,
        eta: float,
        stdNoise: float,
        gamma: float,
        predHorizon: int,
        bandit: Bandit,
    ) -> None:
        """
        Init to the explorer class. Needs
            - eta: float: weighting parameter
            - stdNoise: float: Noise level of the plant
            - gamma: float: input power bound
            - predHorizon: int: prediction horizon in optimization
            - bandit: Bandit: structure with all systems inside
        """
        # Init
        self.nOpts = bandit.nOpts
        self.belief = 1 / self.nOpts * np.ones((self.nOpts))
        self.likelihood = 1 / self.nOpts * np.ones((self.nOpts))
        self.stdNoise = stdNoise
        self.bandit = bandit
        self.gamma = gamma
        self.bound = 2
        self.eta = eta

        self.horizon = predHorizon
        self.rng = np.random.default_rng(seed=24)

    def updateBelief(self, xpMeas: np.ndarray, xMeas: np.ndarray, uMeas: np.ndarray):
        """
        Updates the current belief based on exponetial weighting
        Inputs:
            - xpMeas: State measurement a time k+1, x(k+1)
            - xMead: State measurement a time k, x(k)
            - uMeas: Input u(k)
        """

        res = self.bandit.calcRes(xpMeas, xMeas, uMeas, self.horizon)
        xi = np.ones(self.nOpts)

        # Exponential Weighting
        for ii in range(self.nOpts):
            for tt in range(self.horizon):
                xi[ii] *= np.exp(
                    -self.eta
                    * self.stdNoise ** (-1)
                    * res[:, tt, ii].transpose()
                    @ res[:, tt, ii]
                )

        self.belief = (self.belief * xi) / (self.belief * xi).sum()

    def updateLikelihood(
        self, xpMeas: np.ndarray, xMeas: np.ndarray, uMeas: np.ndarray
    ):
        """
        Helper Function that is not required for input design. Updates the Likelihood
        of the data given each candidate system (required for plotting)
        Inputs:
            - xpMeas: State measurement a time k+1, x(k+1)
            - xMead: State measurement a time k, x(k)
            - uMeas: Input u(k)
        """

        res = self.bandit.calcRes(xpMeas, xMeas, uMeas, self.horizon)
        xi = np.ones(self.nOpts)

        # Exponential Weighting
        for ii in range(self.nOpts):
            for tt in range(self.horizon):
                xi[ii] *= np.exp(
                    -1
                    / 2
                    * self.stdNoise ** (-1)
                    * res[:, tt, ii].transpose()
                    @ res[:, tt, ii]
                )

        self.likelihood = (self.likelihood * xi) / (self.likelihood * xi).sum()

    def randomInput(self):
        """
        Generates a random input (satisfying the gamma-constraint) of suitable length
        """
        u = np.zeros((self.bandit.n_u, self.horizon))
        u[:, :] = np.random.normal(
            loc=0.0,
            scale=self.gamma / np.sqrt(self.bandit.n_u),
            size=(self.bandit.n_u, self.horizon),
        )

        return u

    def inputPolicy(self, xMeas: np.ndarray, rho: float):
        """
        Computes optimal input as proposed in the paper
        Input:
            - xMeas: Current State-measurement x(k)
            - rho: weight parameter rho_k
        """
        arm = self.rng.choice(np.arange(0, self.nOpts), p=self.belief)
        [u_traj_opt, u_vec] = self.optInput(xMeas, self.bandit.systems[arm])
        u_traj_rand = self.randomInput()
        if rho > 1:
            rho = 1
        u_alg = np.sqrt(1 - rho) * u_traj_opt + np.sqrt(rho) * u_traj_rand

        return u_alg, arm, u_vec

    def inputPolicyOracle(self, xMeas: np.ndarray):
        """
        Computes optimal input using oracle knowledge
        Input:
            - xMeas: Current State-measurement x(k)
        """
        arm = 0
        [u_traj, u_vec] = self.optInput(xMeas, self.bandit.systems[arm])
        return u_traj, arm, u_vec

    def optInput(self, x: np.ndarray, system: LTI_discrete):
        """
        Optimal Input Design objective
        Inputs:
            - x: Current state
            - system: System (estimate) to solve the optimization problem with
        """

        # Define Constraints
        cons = (
            {"type": "ineq", "fun": lambda z_opt: z_opt[0]},
            {
                "type": "ineq",
                "fun": lambda z_opt: self.constr_artifical(z_opt, x, system),
            },
            {
                "type": "ineq",
                "fun": lambda z_opt: -z_opt[1:].transpose() @ z_opt[1:]
                + self.gamma**2 * self.horizon,
            },
        )

        #  Define Initial guess for optimization
        x0 = (
            self.gamma
            / np.sqrt(self.bandit.n_u)
            * np.ones(self.bandit.n_u * self.horizon + 1)
        )
        x0[0] = 0

        # Solve for optimal input
        sol = minimize(
            fun=self.optFun,
            x0=x0,
            method="SLSQP",
            constraints=cons,
        )

        u_vec = sol.x[1:]
        u_traj = u_vec.reshape((self.horizon, self.bandit.n_u)).transpose()
        return u_traj, u_vec

    def constr_artifical(self, z_opt: np.ndarray, x: np.ndarray, system: LTI_discrete):
        """
        Sets up the constraint
        - z_opt: Optimization variable
        - x: state x(k)
        - system: System Estimate
        """
        DeltaA = np.zeros((self.nOpts - 1, self.bandit.n_x, self.bandit.n_x))
        DeltaB = np.zeros((self.nOpts - 1, self.bandit.n_x, self.bandit.n_u))

        index = 0
        for jj, sys_comp in enumerate(self.bandit.systems):
            if np.any(system.A - sys_comp.A != 0) or np.any(system.B - sys_comp.B != 0):
                DeltaA[index, :, :] = system.A - sys_comp.A
                DeltaB[index, :, :] = system.B - sys_comp.B
                index += 1

        x_traj = np.zeros((self.bandit.n_x, self.horizon + 1))
        x_traj[:, [0]] = x

        u_vec = z_opt[1:]
        xi = z_opt[0]

        # Reshape u
        u = u_vec.reshape((self.horizon, self.bandit.n_u)).transpose()

        # TODO: Bring to the same format as in the paper

        x_traj[:, 1:] = system.sim(
            x=x, u=u, w=np.zeros((self.bandit.n_x, self.horizon)), tSim=self.horizon
        )
        fun = -xi * np.ones(self.nOpts - 1)

        for jj in range(self.nOpts - 1):
            for tt in range(self.horizon):
                fun[jj] += (
                    1
                    / self.stdNoise
                    * (
                        DeltaA[jj, :, :] @ x_traj[:, tt] + DeltaB[jj, :, :] @ u[:, tt]
                    ).transpose()
                    @ (DeltaA[jj, :, :] @ x_traj[:, tt] + DeltaB[jj, :, :] @ u[:, tt])
                )

        return fun

    def optFun(self, z_opt):
        return -z_opt[0]

    def resetBelief(self):
        """
        Resets the Belief of the explorer instance
        """
        self.belief = 1 / self.nOpts * np.ones((self.nOpts))
        self.likelihood = 1 / self.nOpts * np.ones((self.nOpts))

    def optFun_wagenmaker(self, U_flat, system, cov_x):
        """
        Optimization objective as in [27] for comparison

        """
        k = np.floor(self.horizon / 3).astype(int)
        U_matrix = U_flat[0 : self.bandit.n_u * k].reshape(
            (self.bandit.n_u, k)
        ) + 1j * U_flat[self.bandit.n_u * k :].reshape((self.bandit.n_u, k))
        # Initialize futureCov as a zero matrix
        futureCov = np.zeros((self.bandit.n_x, self.bandit.n_x), dtype=complex)

        # Loop over ell in [1, k]
        for ell in range(1, k + 1):
            z = np.exp(1j * 2 * np.pi * (ell - 1) / k)
            F_ell = np.linalg.inv(z * np.eye(self.bandit.n_x) - system.A)

            U_ell = U_matrix[:, [ell - 1]] @ U_matrix[:, [ell - 1]].conj().T
            term = F_ell @ system.B @ U_ell @ system.B.conj().T @ F_ell.conj().T
            futureCov += (1 / (self.gamma**2 * k**2)) * term

        # Compute final covariance
        covAll = self.gamma**2 * self.horizon * futureCov + cov_x
        return -np.min(np.real(np.linalg.eigvals(covAll)))

    def constr_wagenmaker(self, U_flat):
        k = np.floor(self.horizon / 3).astype(int)
        U_matrix = U_flat[0 : self.bandit.n_u * k].reshape(
            (self.bandit.n_u, k)
        ) + 1j * U_flat[self.bandit.n_u * k :].reshape((self.bandit.n_u, k))
        constr = +((self.gamma) ** 2) * k**2
        for jj in range(0, k):
            constr -= np.trace(U_matrix[:, [jj]] @ U_matrix[:, [jj]].conj().T)
        return np.real(constr)

    def iff_sequence(self, u_f, T):
        """
        Convert Fourier coefficients u_f to a time-domain signal of length T.

        Parameters:
            u_f : ndarray of shape (n_u, k)
                Fourier coefficients for each input channel.
            T : int
                Desired length of the output time-domain signal.

        Returns:
            u_t : ndarray of shape (n_u, T)
                Time-domain signal of length T.
        """
        k = u_f.shape[1]
        u = np.zeros((self.bandit.n_u, k))

        # Perform IFFT for each input channel
        for i in range(self.bandit.n_u):
            u[i, :] = np.real(np.fft.ifft(u_f[i, :]))

        # Repeat the sequence until reaching length T
        repeated_u = np.tile(u, (1, int(np.ceil(T / k))))
        u_t = repeated_u[:, :T]

        return u_t

    def optInput_wagenmaker(self, system: LTI_discrete, cov_x: np.ndarray):
        k = np.floor(self.horizon / 3).astype(int)
        U_0_real = 1 * np.ones((self.bandit.n_u, k)).flatten()
        U_0_imag = np.zeros((self.bandit.n_u, k)).flatten()
        U_0 = np.concatenate([U_0_real, U_0_imag])
        objective_wagenmaker = lambda U: self.optFun_wagenmaker(U, system, cov_x)
        constraints = [{"type": "ineq", "fun": self.constr_wagenmaker}]

        sol = minimize(
            fun=objective_wagenmaker, x0=U_0, method="SLSQP", constraints=constraints
        )

        U_flat = sol.x
        U_matrix = U_flat[0 : self.bandit.n_u * k].reshape(
            (self.bandit.n_u, k)
        ) + 1j * U_flat[self.bandit.n_u * k :].reshape((self.bandit.n_u, k))
        U_opt_time = self.iff_sequence(U_matrix, T=self.horizon)
        U_time = (1 / np.sqrt(2)) * U_opt_time + np.random.normal(
            loc=0.0,
            scale=self.gamma / np.sqrt(2 * self.bandit.n_u),
            size=(self.bandit.n_u, self.horizon),
        )
        return U_time

    def inputWagenmaker(self, cov_x):
        # arm = self.rng.choice(np.arange(0, self.nOpts), p=self.belief)
        arm = 0
        u_traj = self.optInput_wagenmaker(self.bandit.systems[arm], cov_x)
        return u_traj, arm
