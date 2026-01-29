__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "26/01/29"

import numpy as np
import os
from tqdm import tqdm

from fcn.LTI import LTI_discrete
from fcn.bandit import Bandit
from fcn.explorer import Explorer
from fcn.simAndReact import simAndReact

if __name__ == "__main__":
    # Specify Problem Setup
    n_u = 2
    n_x = 3
    nSys = 20
    nEpochs = 5
    x0 = np.array([[0], [0], [0]])
    predHorizon = 15
    gamma = 1
    nMC = 30  # 100 Reduced for improved runtime

    eta = 0.1

    stdNoise = 1
    noiseGen = np.random.default_rng(seed=42)
    sysGen = np.random.default_rng(seed=7)


    # True system
    A_true = np.array([[0, 0.1, 0], [0, 0, 0], [0, 0, 0.9]])
    B_true = np.array([[0, 0], [1, 0], [0, 1]])

    A_candidates = np.zeros((n_x, n_x, nSys))

    plant = LTI_discrete(A=A_true, B=B_true)

    # Set of systems
    sysSet = [plant]

    for i in range(nSys - 1):
        sysSet.append(
            LTI_discrete(
                A=A_true + 0.5 * (sysGen.random((n_x, n_x)) - 0.5),
                B=B_true + (sysGen.random((n_x, n_u)) - 0.5),
            )
        )

    # Init rho sequences
    nRho = 3
    rho1 = 1 / (1 + np.arange(nEpochs))
    rho2 = 1 / (1 + np.arange(nEpochs)) ** 2
    rhoCE = np.zeros_like(rho1)

    rho = [rho1, rho2, rhoCE]

    # Initialize Bandit
    bandit = Bandit(systems=sysSet)

    # Initialize Explorers
    explorer = Explorer(
        eta=eta, stdNoise=stdNoise, gamma=gamma, predHorizon=predHorizon, bandit=bandit
    )

    explorer_oracle = Explorer(
        eta=eta,
        stdNoise=stdNoise,
        gamma=gamma,
        predHorizon=predHorizon * nEpochs,
        bandit=bandit,
    )

    # Init data vectors
    x_traj = np.zeros((n_x, nEpochs * predHorizon + 1, nMC, nRho + 3))
    u_traj = np.zeros((n_u, nEpochs * predHorizon, nMC, nRho + 3))
    belief_traj = np.zeros((nSys, nEpochs + 1, nMC, nRho + 3))
    likelihood_traj = np.zeros((nSys, nEpochs + 1, nMC, nRho + 3))

    arm_traj = np.zeros((nEpochs, nMC, nRho + 3))
    rho_traj = np.zeros((nEpochs, nMC, nRho + 3))


    # Monte carlo Simulations
    for jMC in tqdm(range(nMC)):
        # Generate noise for this MC sim
        w = noiseGen.normal(0, stdNoise, (n_x, nEpochs * predHorizon))

        for jRho in range(nRho):
            (
                x_traj[:, :, jMC, jRho],
                u_traj[:, :, jMC, jRho],
                belief_traj[:, :, jMC, jRho],
                likelihood_traj[:, :, jMC, jRho],
                arm_traj[:, jMC, jRho],
                rho_traj[:, jMC, jRho],
            ) = simAndReact(
                plant=plant,
                explorer=explorer,
                x0=x0,
                w=w,
                nEpochs=nEpochs,
                rho=rho[jRho],
                mode=1,
            )

        # Sim and react algorithm with Wagenmaker
        (
            x_traj[:, :, jMC, -3],
            u_traj[:, :, jMC, -3],
            belief_traj[:, :, jMC, -3],
            likelihood_traj[:, :, jMC, -3],
            arm_traj[:, jMC, -3],
            rho_traj[:, jMC, -3],
        ) = simAndReact(
            plant=plant,
            explorer=explorer,
            x0=x0,
            w=w,
            nEpochs=nEpochs,
            rho=np.zeros(nEpochs),
            mode=4,
        )

        # Sim and react algorithm with oracle
        (
            x_traj[:, :, jMC, -2],
            u_traj[:, :, jMC, -2],
            belief_traj[:, :, jMC, -2],
            likelihood_traj[:, :, jMC, -2],
            arm_traj[:, jMC, -2],
            rho_traj[:, jMC, -2],
        ) = simAndReact(
            plant=plant,
            explorer=explorer,
            x0=x0,
            w=w,
            nEpochs=nEpochs,
            rho=np.zeros(nEpochs),
            mode=2,
        )

        # Sim and react isotropic Noise
        (
            x_traj[:, :, jMC, -1],
            u_traj[:, :, jMC, -1],
            belief_traj[:, :, jMC, -1],
            likelihood_traj[:, :, jMC, -1],
            arm_traj[:, jMC, -1],
            rho_traj[:, jMC, -1],
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
        "data/exp_mainExample_randomSet",
        sysSet=sysSet,
        nMC=nMC,
        x_traj=x_traj,
        u_traj=u_traj,
        belief_traj=belief_traj,
        likelihood_traj=likelihood_traj,
        arm_traj=arm_traj,
    )
