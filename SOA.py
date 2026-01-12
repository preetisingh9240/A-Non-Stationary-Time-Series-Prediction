import numpy as np
import time


#  Starfish Optimization Algorithm (SOA)
def SOA(Xpos, fobj, lb, ub, Max_it):
    GP = 0.5  # parameter
    Npop, nD = Xpos.shape
    if np.isscalar(lb):
        lb = np.full(nD, lb)
    if np.isscalar(ub):
        ub = np.full(nD, ub)

    fvalbest = float('inf')
    Curve = np.zeros(Max_it)
    Fitness = np.array([fobj(ind) for ind in Xpos])

    order = np.argmin(Fitness)
    fvalbest = Fitness[order]
    xposbest = Xpos[order].copy()

    ct = time.time()
    T = 0
    while T < Max_it:
        theta = np.pi / 2 * T / Max_it
        tEO = (Max_it - T) / Max_it * np.cos(theta)
        newX = np.zeros_like(Xpos)

        if np.random.rand() < GP:  # Exploration
            for i in range(Npop):
                if nD > 5:
                    jp1 = np.random.choice(nD, 5, replace=False)
                    for j in jp1:
                        pm = (2 * np.random.rand() - 1) * np.pi
                        r = np.random.rand()
                        if r < GP:
                            newX[i, j] = Xpos[i, j] + pm * (xposbest[j] - Xpos[i, j]) * np.cos(theta)
                        else:
                            newX[i, j] = Xpos[i, j] - pm * (xposbest[j] - Xpos[i, j]) * np.sin(theta)
                        if newX[i, j] > ub[j] or newX[i, j] < lb[j]:
                            newX[i, j] = Xpos[i, j]
                else:
                    jp2 = int(nD * np.random.rand())
                    im = np.random.permutation(Npop)
                    rand1, rand2 = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
                    newX[i, jp2] = tEO * Xpos[i, jp2] + rand1 * (Xpos[im[0], jp2] - Xpos[i, jp2]) + rand2 * (
                            Xpos[im[1], jp2] - Xpos[i, jp2])
                    if newX[i, jp2] > ub[i, jp2] or newX[i, jp2] < lb[i, jp2]:
                        newX[i, jp2] = Xpos[i, jp2]
                newX[i] = np.clip(newX[i], lb[i], ub[i])
        else:  # Exploitation
            df = np.random.permutation(Npop)[:5]
            dm = np.array([xposbest - Xpos[df[i]] for i in range(5)])
            for i in range(Npop):
                r1, r2 = np.random.rand(), np.random.rand()
                kp = np.random.choice(5, 2, replace=False)
                newX[i] = Xpos[i] + r1 * dm[kp[0]] + r2 * dm[kp[1]]
                if i == Npop - 1:
                    newX[i] = np.exp(-T * Npop / Max_it) * Xpos[i]
                newX[i] = np.clip(newX[i], lb[i], ub[i])

        # Evaluate and update
        for i in range(Npop):
            newFit = fobj(newX[i])
            if newFit < Fitness[i]:
                Fitness[i] = newFit
                Xpos[i] = newX[i].copy()
                if newFit < fvalbest:
                    fvalbest = newFit
                    xposbest = newX[i].copy()

        Curve[T] = fvalbest
        T += 1
    ct = time.time() - ct
    return fvalbest, Curve, xposbest, ct


