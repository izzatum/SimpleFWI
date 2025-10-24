import numpy as np
from scipy import sparse
from pylops import LinearOperator
from pylops.signalprocessing import Bilinear


def getA(f, m, h, n):
    """Construct Helmholtz matrix (omega*m)**2 + Laplacian

    Parameters
    ----------
    f : :obj:`numpy.ndarray`
        1-D array of frequencies
    m : :obj:`numpy.ndarray`
        Vector of model parameters (slowness-squared model)
    h : :obj:`numpy.ndarray`
        1-D array of model spacing
    n : :obj:`numpy.ndarray`
        1-D array of number of elements per dimension (nz, nx)


    Returns
    -------
    : :obj:`scipy.sparse.csr`
        sparse compressed Helmholtz matrix with dimension of N x N
        N - total number of model parameters

    """
    N = np.prod(n)
    omega = 1.0e-3 * 2.0 * np.pi * f
    m = m.ravel()

    # derivative operator D'' = D'.T @ D'
    d1 = np.repeat([[-1.0], [1.0]], n[0], axis=1) / h[0]
    d2 = np.repeat([[-1.0], [1.0]], n[1], axis=1) / h[1]
    diags_pos = np.array([0, 1])

    D1 = sparse.spdiags(d1, diags_pos, n[0] - 1, n[0], format="csc")
    D2 = sparse.spdiags(d2, diags_pos, n[1] - 1, n[1], format="csc")

    # Laplacian
    # L = - sparse.kron(D2.T @ D2, sparse.eye(n[0]), format='csc') - \
    #    sparse.kron(sparse.eye(n[1]), D1.T @ D1, format='csc')
    L = -sparse.kron(D1.T @ D1, sparse.eye(n[1]), format="csc") - sparse.kron(
        sparse.eye(n[0]), D2.T @ D2, format="csc"
    )

    a = np.ones(n)
    a[:, [0, -1]] = 0.5
    a[[0, -1], :] = 0.5
    a = a.ravel()

    return (
        omega**2 * sparse.spdiags(a * m, 0, N, N, format="csc")
        + (2.0j * omega / h[0]) * sparse.spdiags((1.0 - a) * np.sqrt(m), 0, N, N, format="csc")
        + L
    )


def getP(h, xs, zs, x, z, dtype="float"):
    """Construct sampling matrix

    Parameters
    ----------
    h : :obj:`numpy.ndarray`
        1-D array of model spacing
    xs : :obj:`numpy.ndarray`
        1-D array of sampling points in x
    zs : :obj:`numpy.ndarray`
        1-D array of sampling points in z
    x : :obj:`numpy.ndarray`
        1-D array of x
    z : :obj:`numpy.ndarray`
        1-D array of z
    dtype: :obj:data type


    Returns
    -------
    : :obj:`pylops.LinearOperator`
        sampling matrix
    """
    ixs = np.array(xs) / h[1]
    izs = np.array(zs) / h[0]
    nx, nz = len(x), len(z)

    return Bilinear(np.vstack((izs, ixs)), (nz, nx), dtype=dtype)


def getG(f, m, u, h, n):
    """Construct Jacobian matrix of Helmholtz operator G(m, u) = d(A(m)*u)/dm

    Parameters
    ----------
    f : :obj:`numpy.ndarray`
        1-D array of frequencies
    m : :obj:`numpy.ndarray`
        Vector of model parameters (slowness-squared model)
    u : :obj:`numpy.ndarray`
        2-D array of wavefields with dimension of N x Ns
        N - total number of model parameters
        Ns - total number of sources
    h : :obj:`numpy.ndarray`
        1-D array of model spacing
    n : :obj:`numpy.ndarray`
        1-D array of number of elements per dimension (nz, nx)


    Returns
    -------
    : :obj:`numpy.ndarray`
        2-D array of G(m, u) = d(A(m)*u)/dm with dimension N x Ns
    """

    omega = 1.0e-3 * 2.0 * np.pi * f

    a = np.ones(n)
    a[:, [0, -1]] = 0.5
    a[[0, -1], :] = 0.5
    a = a.ravel()
    m = m.ravel()
    u = u.ravel()

    return omega**2 * sparse.diags(a * u) + (2.0j * omega / h[0]) * sparse.diags(
        0.5 * (1.0 - a) * u / np.sqrt(m)
    )


class JacobianForwardSolver(LinearOperator):
    r"""Jacobian operator of forward solver

    Applies to data difference to compute gradient of data misfit.

    This operator also can be used to construct the data misfit Hessian operator.

    Parameters
    ----------
    m : :obj:`numpy.ndarray`
        Vector of model parameters (slowness-squared model)
    U : :obj:`numpy.ndarray`
        3-D array of wavefields with dimension of N x Ns x Nf
        N - total number of model parameters
        Ns - total number of sources
        Nf - total number of frequencies
    model : :obj:`dict`
        Dictionary of parameters for modelling
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    """

    def __init__(self, m, U, model, dtype="complex"):
        super().__init__()
        self.U = U
        self.nr = len(model["zr"])
        self.ns = len(model["zs"])
        self.nf = len(model["f"])
        self.N = np.prod(model["n"])

        self.f = model["f"]
        self.n = model["n"]
        self.h = model["h"]

        self.shape = (self.nr * self.ns * self.nf, self.N)
        self.dtype = np.dtype(dtype)
        self.explicit = False

        self.Pr = getP(model["h"], model["xr"], model["zr"], model["x"], model["z"], dtype)
        self.Gk = lambda fr, u: getG(fr, m, u, self.h, self.n)
        self.Ak = lambda fr: getA(fr, m, self.h, self.n)
        self.Aks = [sparse.linalg.factorized(self.Ak(freq)) for freq in self.f]
        self.AHks = [sparse.linalg.factorized(self.Ak(freq).H) for freq in self.f]

    def _matvec(self, v):

        y = []
        for i, freq in enumerate(self.f):
            Rk = np.zeros((self.N, self.ns), dtype="complex")
            for s in range(self.ns):
                Rk[:, s] = (-self.Gk(freq, self.U[..., s, i]) * v.reshape(-1, 1)).squeeze()
            ARk = self.Aks[i](Rk)
            # ARk = sparse.linalg.spsolve(self.Ak(freq), Rk)
            y.append(self.Pr @ ARk)
        return np.vstack(y).reshape(-1, 1)

    def _rmatvec(self, v):
        y = 0.0
        v = v.reshape((self.nr, self.ns, self.nf))
        for i, freq in enumerate(self.f):
            Pv = self.Pr.T * v[..., i]
            Rk = self.AHks[i](Pv)
            # Rk = sparse.linalg.spsolve(self.Ak(freq).H, Pv)
            # imaging condition
            # GRk = self.Gk(freq, self.U[..., i]).conj() * Rk
            for s in range(self.ns):
                y -= self.Gk(freq, self.U[..., s, i]).H * Rk[..., s]
                # y -= GRk[..., s].reshape(-1, 1)
        return y.reshape(-1, 1)


class ForwardSolver:
    r"""Forward solver for frequency-domain FWI

    Maps model parameters m to data D:
    R^{N} -> C^{Nr x Ns x Nf}
    N - total number of model parameters
    Nr - total number of receivers
    Ns - total number of sources
    Nf - total number of frequencies

    Parameters
    ----------
    model : :obj:`dict`
        Dictionary of parameters for modelling

    """

    def __init__(self, model):
        self.model = model
        self.nr = len(model["zr"])
        self.ns = len(model["zs"])
        self.nf = len(model["f"])
        self.nxz = np.prod(model["n"])

        self.f = model["f"]
        self.n = model["n"]
        self.h = model["h"]

        self.Ps = getP(
            model["h"], model["xs"], model["zs"], model["x"], model["z"], dtype="complex"
        )
        self.Pr = getP(
            model["h"], model["xr"], model["zr"], model["x"], model["z"], dtype="complex"
        )
        self.Q = self.Ps.T @ model["q"]

    def solve(self, m, rsample=True):
        """Solves the Helmholtz equation and maps the solution to data

        Parameters
        ----------
        m : :obj:`numpy.ndarray`
            Vector of model parameters (slowness-squared model)
        rsample : :obj:`bool`
            Sample at receivers or return full field

        Returns
        -------
        D : :obj:`numpy.ndarray`
            Vector of computed data with flattened dimension of Nr x Ns x Nf
        DF: :obj:`pylops.LinearOperator`
            Linear operator of Jacobian of forward solver
        """
        U = np.zeros((self.nxz, self.ns, self.nf), dtype="complex")
        D = np.zeros((self.nr, self.ns, self.nf), dtype="complex")

        for i, f in enumerate(self.f):
            Ai = getA(f, m, self.h, self.n)
            Aifact = sparse.linalg.factorized(Ai)
            U[:, :, i] = (Aifact(self.Q)).reshape(self.nxz, self.ns)
            # U[:, :, i] = sparse.linalg.spsolve(Ai, self.Q).reshape(self.nxz, self.ns)
            # iterative solver, but can only do one source at the time!
            # U[:, :, i] = sparse.linalg.gmres(Ai, self.Q)[0].reshape(self.nxz, self.ns)
            D[:, :, i] = (self.Pr @ U[..., i]).reshape(self.nr, self.ns)

        DF = JacobianForwardSolver(m, U, self.model)

        if rsample:
            return D.reshape(-1, 1), DF
        else:
            return U, DF
