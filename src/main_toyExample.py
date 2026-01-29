__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "26/01/21"

import numpy as np
import os
from tqdm import tqdm

from fcn.LTI import LTI_discrete
from fcn.bandit import Bandit
from fcn.explorer import Explorer
from fcn.simAndReact import simAndReact

if __name__ == "__main__":

    # Specify Problem Setup
    n_u = 5
    n_x = 6
    nSys = 2
    nEpochs = 40
    x0 = np.array([[0], [0], [0], [0], [0], [0]])
    predHorizon = 10
    gamma = 1
    nMC = 20

    eta = 0.1

    stdNoise = 0.1
    noiseGen = np.random.default_rng()

    # True system
    A_true = np.array(
        [
            [0, 0.1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0.9, 0, 0, 0],
            [0, 0, 0, 0.9, 0, 0],
            [0, 0, 0, 0, 0.9, 0],
            [0, 0, 0, 0, 0, 0.9],
        ]
    )
    B_true = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )

    A_false = np.array(
        [
            [0, 0.2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0.9, 0, 0, 0],
            [0, 0, 0, 0.9, 0, 0],
            [0, 0, 0, 0, 0.9, 0],
            [0, 0, 0, 0, 0, 0.9],
        ]
    )
    B_false = B_true

    plant = LTI_discrete(A=A_true, B=B_true)
    plant_false = LTI_discrete(A=A_false, B=B_false)

    # Set of systems
    sysSet = [plant, plant_false]

    # Initialize Bandit
    bandit = Bandit(systems=sysSet)

    # Initialize Explorer
    explorer = Explorer(
        eta=eta, stdNoise=stdNoise, gamma=gamma, predHorizon=predHorizon, bandit=bandit
    )

    # Init data vectors
    x_traj = np.zeros((n_x, nEpochs * predHorizon + 1, nMC, 3))
    u_traj = np.zeros((n_u, nEpochs * predHorizon, nMC, 3))
    belief_traj = np.zeros((nSys, nEpochs + 1, nMC, 3))
    likelihood_traj = np.zeros((nSys, nEpochs + 1, nMC, 3))

    arm_traj = np.zeros((nEpochs, nMC, 3))
    rho_traj = np.zeros((nEpochs, nMC, 3))

    # Monte carlo Simulations
    for jMC in tqdm(range(nMC)):
        # Generate noise for this MC sim
        w = noiseGen.normal(0, stdNoise, (n_x, nEpochs * predHorizon))

        # Sim and react Algo with CE
        (
            x_traj[:, :, jMC, 0],
            u_traj[:, :, jMC, 0],
            belief_traj[:, :, jMC, 0],
            likelihood_traj[:, :, jMC, 0],
            arm_traj[:, jMC, 0],
            rho_traj[:, jMC, 0],
        ) = simAndReact(
            plant=plant,
            explorer=explorer,
            x0=x0,
            w=w,
            nEpochs=nEpochs,
            rho=np.zeros(nEpochs),
            mode=1,
        )

        # Sim and react Oracle
        (
            x_traj[:, :, jMC, 1],
            u_traj[:, :, jMC, 1],
            belief_traj[:, :, jMC, 1],
            likelihood_traj[:, :, jMC, 1],
            arm_traj[:, jMC, 1],
            rho_traj[:, jMC, 1],
        ) = simAndReact(
            plant=plant,
            explorer=explorer,
            x0=x0,
            w=w,
            nEpochs=nEpochs,
            rho=np.zeros(nEpochs),
            mode=2,
        )

        # Sim and react Isotrpoic Gaussian Inputs
        (
            x_traj[:, :, jMC, 2],
            u_traj[:, :, jMC, 2],
            belief_traj[:, :, jMC, 2],
            likelihood_traj[:, :, jMC, 2],
            arm_traj[:, jMC, 2],
            rho_traj[:, jMC, 2],
        ) = simAndReact(
            plant=plant,
            explorer=explorer,
            x0=x0,
            w=w,
            nEpochs=nEpochs,
            rho=np.zeros(nEpochs),
            mode=3,
        )

    #  Save data
    if not (os.path.isdir("./data")):
        os.mkdir("./data")

    # For plots in Python
    np.savez(
        "./data/exp_toyExample.npz",
        sysSet=sysSet,
        nMC=nMC,
        x_traj=x_traj,
        u_traj=u_traj,
        belief_traj=belief_traj,
        likelihood_traj=likelihood_traj,
        arm_traj=arm_traj,
    )
