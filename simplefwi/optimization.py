import numpy as np


def BBiter(misfitfunc, m0, tol=1.e-6, maxit=5, bounds=None):
    mk = m0.copy()
    fk, gk, _ = misfitfunc.evaluate(mk)
    tk = np.linalg.norm(gk)
    history = []

    print(f"Optimization starts!")

    for k in range(1, maxit + 1):
        # update
        sk = -gk / tk
        mk += sk

        # apply bounds
        if bounds is not None:
            mk = np.maximum(mk, bounds[0])
            mk = np.minimum(mk, bounds[1])

        # gradient update
        fk, gk, _ = misfitfunc.evaluate(mk)

        # update step-length
        tk += (sk.T @ gk) / np.linalg.norm(sk) ** 2

        print(f"k: {k}  fk: {fk:.2f}  ||gk||: {np.linalg.norm(gk):.2f}")
        history.append([k, fk, np.linalg.norm(gk)])

        if fk < tol:
            print(f"Optimization ended!")
            break
    print(f"Optimization ended!")

    return np.vstack(history), mk, gk


def CGiter(misfitfunc, m0, Dobs, forwardSolver, tol=1.e-6, maxit=5, bounds=None):
    mk = m0.copy()
    fk0, gk0, _ = misfitfunc.evaluate(mk)
    dk = -gk0
    history = []

    print(f"Optimization starts!")

    for k in range(1, maxit + 1):
        # compute deterministic step-size
        alpha = deterministicAlpha(mk, dk, Dobs, forwardSolver)

        # update
        mk += alpha * dk

        # apply bounds
        if bounds is not None:
            mk = np.maximum(mk, bounds[0])
            mk = np.minimum(mk, bounds[1])

        # gradient update
        fk, gk, _ = misfitfunc.evaluate(mk)

        print(f"k: {k}  fk: {fk:.2f}  ||gk||: {np.linalg.norm(gk):.2f}")
        history.append([k, fk, np.linalg.norm(gk)])

        if fk < tol:
            print(f"Optimization ended!")
            break

        denom = dk.T @ (gk - gk0)
        beta1 = gk.T @ (gk - gk0) / denom
        beta2 = gk.T @ gk / denom
        beta = np.maximum(0, np.minimum(beta1, beta2))

        dk = -gk + beta * dk
        gk0 = gk

    print(f"Optimization ended!")

    return np.vstack(history), mk, gk


def deterministicAlpha(mk, dk, Dobs, forwardSolver):
    # compute epsilon
    dkmax = np.maximum(0, max(dk))
    mmax = np.maximum(0, max(mk))

    epsilon = (0.01 * mmax) / (dkmax + 1.e-6)

    # prepare data for step-size
    mtemp = mk + epsilon * dk
    Dsim, _ = forwardSolver.solve(mk)
    Dtemp, _ = forwardSolver.solve(mtemp)

    # compute step-size
    res = Dsim - Dobs
    dtemp = Dtemp - Dsim

    a1 = -dtemp.conj().T @ res
    a2 = dtemp.conj().T @ dtemp
    alpha = (a1 * epsilon) / (a2 + 1.e-6)

    return alpha.real
