__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "25/08/15"

from fcn.explorer import Explorer
from fcn.LTI import LTI_discrete
import numpy as np


def simAndReact(
    plant: LTI_discrete,
    explorer: Explorer,
    x0: np.ndarray,
    w: np.ndarray,
    nEpochs: int,
    rho: np.ndarray,
    mode: int,
):
    """
    Helper Function that carries out the simulation
    Inputs:
        - plant: true system to be simulated
        - explorer: instance of input genration class Explorer
        - x0: Initial condition for simulation
        - w: array with noise vector over simulation horizon
        - nEpochs: number of epochs to be simulated
        - rho: sequence of parameter rho_k
        - mode: integer,
            1= input policy presented in paper
            2= input policy presented in paper with oracle knowledge
            3= random inputs
            4= input policy described in [27] with oracle knowledge
    """

    # Reset belief at start of simulation
    explorer.resetBelief()

    # Init data vectors
    x_traj = np.zeros((explorer.bandit.n_x, nEpochs * explorer.horizon + 1))
    x_traj[:, [0]] = x0
    u_traj = np.zeros((explorer.bandit.n_u, nEpochs * explorer.horizon))
    belief_traj = np.zeros((explorer.bandit.nOpts, nEpochs + 1))
    belief_traj[:, 0] = explorer.belief
    likelihood_traj = belief_traj
    arm_traj = np.zeros((nEpochs))

    # Sim and react
    for j in range(nEpochs):
        x = x_traj[:, [j * explorer.horizon]]
        # select mode for input generation
        if mode == 1:
            u, arm, u_vec = explorer.inputPolicy(xMeas=x, rho=rho[j])
        elif mode == 2:
            u, arm, u_vec = explorer.inputPolicyOracle(xMeas=x)
        elif mode == 3:
            u = explorer.randomInput()
            arm = -1
        elif mode == 4:
            u, arm = explorer.inputWagenmaker(cov_x=x_traj @ x_traj.T)

        # Sim with given inputs
        x_p = plant.sim(
            x=x,
            u=u,
            w=w[:, j * explorer.horizon : (j + 1) * explorer.horizon],
            tSim=explorer.horizon,
        )

        # update data vectors
        x_traj[:, j * explorer.horizon + 1 : (j + 1) * explorer.horizon + 1] = x_p
        u_traj[:, j * explorer.horizon : (j + 1) * explorer.horizon] = u
        
        # update belief
        explorer.updateBelief(
            xpMeas=x_traj[:, j * explorer.horizon + 1 : (j + 1) * explorer.horizon + 1],
            xMeas=x_traj[:, j * explorer.horizon : (j + 1) * explorer.horizon],
            uMeas=u,
        )

        # update likelihood (for plotting)
        explorer.updateLikelihood(
            xpMeas=x_traj[:, j * explorer.horizon + 1 : (j + 1) * explorer.horizon + 1],
            xMeas=x_traj[:, j * explorer.horizon : (j + 1) * explorer.horizon],
            uMeas=u,
        )

        # update data vectors
        belief_traj[:, j + 1] = explorer.belief
        likelihood_traj[:, j + 1] = explorer.likelihood
        arm_traj[j] = arm

    return x_traj, u_traj, belief_traj, likelihood_traj, arm_traj, rho
