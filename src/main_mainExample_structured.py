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
    n_u = 2
    n_x = 3
    nSys = 4
    nEpochs = 5
    x0 = np.array([[0], [0], [0]])
    predHorizon = 15
    gamma = 1
    nMC = 100

    eta = 0.01

    stdNoise = 1
    noiseGen = np.random.default_rng(seed=7)


    # True system
    A_true = np.array([[0, 0.1, 0], [0, 0, 0], [0, 0, 0.9]])
    B_true = np.array([[0, 0], [1, 0], [0, 1]])

    A_candidates = np.zeros((n_x, n_x, nSys))

    A_candidates[:, :, 0] = np.array([[0, 0, 0.1], [0, 0, 0], [0, 0, 0.9]])
    A_candidates[:, :, 1] = np.array([[0, 0, 0.1], [0, 0, 0], [0, 0, 0.8]])
    A_candidates[:, :, 2] = np.array([[0, 0.1, 0], [0, 0, 0], [0, 0, 0.8]])


    plant = LTI_discrete(A=A_true, B=B_true)

    # Set of systems
    sysSet = [plant]

    for i in range(nSys - 1):
    sysSet.append(LTI_discrete(A=A_candidates[:, :, i], B=B_true))

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
    x_traj = np.zeros((n_x, nEpochs * predHorizon + 1, nMC, 5))
    u_traj = np.zeros((n_u, nEpochs * predHorizon, nMC, 5))
    belief_traj = np.zeros((nSys, nEpochs + 1, nMC, 5))
    likelihood_traj = np.zeros((nSys, nEpochs + 1, nMC, 5))


    arm_traj = np.zeros((nEpochs, nMC, 5))
    rho_traj = np.zeros((nEpochs, nMC, 5))


    # Monte carlo Simulations
    for jMC in tqdm(range(nMC)):
    # Generate noise for this MC sim
    w = noiseGen.normal(0, stdNoise, (n_x, nEpochs * predHorizon))

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

    # Sim and react algorithm with oracle
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

    # Sim and react Wagenmaker
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
        mode=4,
    )

    # Sim and react full Oracle planning
    for jEps in range(1, nEpochs + 1):
        explorer_oracle.resetBelief()
        belief_traj[:, 0, jMC, -2] = explorer_oracle.belief
        likelihood_traj[:, 0, jMC, -2] = explorer_oracle.likelihood
        explorer_oracle.horizon = jEps * predHorizon
        (
            x_traj_oracle,
            u_traj_oracle,
            belief_traj_oracle,
            likelihood_traj_oracle,
            _,
            _,
        ) = simAndReact(
            plant=plant,
            explorer=explorer_oracle,
            x0=x0,
            w=w,
            nEpochs=1,
            rho=np.zeros(nEpochs),
            mode=2,
        )

        belief_traj[:, jEps, jMC, -2] = belief_traj_oracle[:, -1]
        likelihood_traj[:, jEps, jMC, -2] = likelihood_traj_oracle[:, -1]

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
    "data/exp_mainExample_structured",
    sysSet=sysSet,
    nMC=nMC,
    x_traj=x_traj,
    u_traj=u_traj,
    belief_traj=belief_traj,
    likelihood_traj=likelihood_traj,
    arm_traj=arm_traj,
    )

