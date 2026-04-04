"""
SimpleFWI Core Module - Rebuilt from Scratch
==============================================

Frequency-domain Full Waveform Inversion using adjoint-state method.

Clean implementation using current package APIs:
- scipy 1.16.2: No .H attribute, use .conj().T
- scipy.sparse.linalg.factorized: Accepts 2D arrays only
- PyLops 2.5.0: Bilinear.T returns 3D arrays

Mathematical foundation:
- Forward: A(ω,m)·u = q  →  d = Pr·u
- Jacobian: J @ dm = Pr·A^{-1}·(-G·dm)
- Adjoint: J^H @ dd = -Re{G^H·(A^{-H}·Pr^H·dd)}

Author: Rebuilt October 2025
"""

import numpy as np
from scipy import sparse
from pylops import LinearOperator
from pylops.signalprocessing import Bilinear


def build_helmholtz_matrix(freq, model_params, grid_spacing, grid_dims):
    """
    Build Helmholtz operator: A(ω,m) = ω²M + 2iωC + L

    Parameters
    ----------
    freq : float
        Frequency in Hz
    model_params : ndarray, shape (N,) or (N,1)
        Slowness-squared model: m = 1/v²
    grid_spacing : array-like, [dz, dx]
        Grid spacing in meters
    grid_dims : array-like, [nz, nx]
        Grid dimensions

    Returns
    -------
    A : scipy.sparse.csc_matrix
        Helmholtz matrix (N, N)
    """
    nz, nx = grid_dims
    dz, dx = grid_spacing
    N = nz * nx

    # Angular frequency (with 1e-3 scaling factor)
    omega = 1e-3 * 2.0 * np.pi * freq

    # Flatten model
    m = np.asarray(model_params).ravel()

    # Build first derivative operators
    # D1: derivative in z-direction
    d1_vals = np.repeat([[-1.0], [1.0]], nz, axis=1) / dz
    D1 = sparse.spdiags(d1_vals, [0, 1], nz - 1, nz, format="csc")

    # D2: derivative in x-direction
    d2_vals = np.repeat([[-1.0], [1.0]], nx, axis=1) / dx
    D2 = sparse.spdiags(d2_vals, [0, 1], nx - 1, nx, format="csc")

    # Build 2D Laplacian: L = -∇² = -(∂²/∂z² + ∂²/∂x²)
    # Using Kronecker product for 2D operators
    L = -sparse.kron(D1.T @ D1, sparse.eye(nx), format="csc") - sparse.kron(
        sparse.eye(nz), D2.T @ D2, format="csc"
    )

    # Edge weights for absorbing boundary condition
    a = np.ones((nz, nx))
    a[:, [0, -1]] = 0.5
    a[[0, -1], :] = 0.5
    a = a.ravel()

    # Helmholtz operator: A = ω²·diag(a·m) + 2iω/dz·diag((1-a)·√m) + L
    A = (
        omega**2 * sparse.spdiags(a * m, 0, N, N, format="csc")
        + (2.0j * omega / dz) * sparse.spdiags((1.0 - a) * np.sqrt(m), 0, N, N, format="csc")
        + L
    )

    return A


def build_interpolation_operator(
    grid_spacing, positions_x, positions_z, coord_x, coord_z, dtype="complex128"
):
    """
    Build bilinear interpolation operator for sampling at arbitrary positions.

    Parameters
    ----------
    grid_spacing : array-like, [dz, dx]
        Grid spacing in meters
    positions_x, positions_z : ndarray
        Physical coordinates of sampling points
    coord_x, coord_z : ndarray
        Grid coordinate vectors
    dtype : str
        Data type

    Returns
    -------
    P : pylops.LinearOperator
        Bilinear interpolation operator
    """
    dz, dx = grid_spacing
    nz, nx = len(coord_z), len(coord_x)

    # Convert physical coordinates to grid indices
    iz = np.asarray(positions_z) / dz
    ix = np.asarray(positions_x) / dx

    # Clamp positions to valid range [0, n-1-eps] to avoid out-of-bounds
    # PyLops Bilinear needs room for interpolation (accesses index+1)
    eps = 1e-5
    iz = np.clip(iz, 0, nz - 1 - eps)
    ix = np.clip(ix, 0, nx - 1 - eps)

    # Stack as (z, x) for PyLops convention
    positions = np.vstack((iz, ix))

    return Bilinear(positions, (nz, nx), dtype=dtype)


def build_sensitivity_kernel(freq, model_params, wavefield, grid_spacing, grid_dims):
    """
    Build sensitivity kernel: G = d(A(m)·u)/dm

    This is a diagonal operator representing how model perturbations
    affect the Helmholtz operator times the wavefield.

    Parameters
    ----------
    freq : float
        Frequency in Hz
    model_params : ndarray, shape (N,)
        Slowness-squared model
    wavefield : ndarray, shape (N,)
        Wavefield (pressure)
    grid_spacing : array-like, [dz, dx]
        Grid spacing
    grid_dims : array-like, [nz, nx]
        Grid dimensions

    Returns
    -------
    G : scipy.sparse.csc_matrix
        Diagonal sensitivity matrix (N, N)
    """
    nz, nx = grid_dims
    dz, dx = grid_spacing

    omega = 1e-3 * 2.0 * np.pi * freq

    # Flatten inputs
    m = np.asarray(model_params).ravel()
    u = np.asarray(wavefield).ravel()

    # Edge weights (must match build_helmholtz_matrix)
    a = np.ones((nz, nx))
    a[:, [0, -1]] = 0.5
    a[[0, -1], :] = 0.5
    a = a.ravel()

    # G = d(A·u)/dm = ω²·diag(a·u) + (2iω/dz)·diag(0.5·(1-a)·u/√m)
    return omega**2 * sparse.diags(a * u) + (2.0j * omega / dz) * sparse.diags(
        0.5 * (1.0 - a) * u / np.sqrt(m)
    )


class JacobianOperator(LinearOperator):
    """
    Matrix-free Jacobian operator for frequency-domain FWI.

    Forward action: J @ dm = Pr · A^{-1} · (-G · dm)
    Adjoint action: J^H @ dd = -Re{sum_freq sum_src G^H · v}
                    where v = A^{-H} · Pr^H · dd

    Parameters
    ----------
    model : ndarray, shape (N, 1)
        Current slowness-squared model
    wavefields : ndarray, shape (N, ns, nf)
        Forward wavefields for all sources and frequencies
    model_dict : dict
        Model parameters dictionary
    """

    def __init__(self, model, wavefields, model_dict):
        # Store dimensions
        self.N = np.prod(model_dict["n"])
        self.ns = len(model_dict["zs"])
        self.nr = len(model_dict["zr"])
        self.nf = len(model_dict["f"])

        # Shape: (nr*ns*nf, N)
        shape = (self.nr * self.ns * self.nf, self.N)
        dtype = np.dtype("complex128")

        # Initialize parent class first
        super().__init__(dtype=dtype, shape=shape)

        self.explicit = False

        # Store model and wavefields
        self.m = np.asarray(model).ravel()
        self.U = wavefields  # Shape: (N, ns, nf)

        # Store model parameters
        self.f = model_dict["f"]
        self.h = model_dict["h"]
        self.n = model_dict["n"]

        # Build interpolation operators
        self.Pr = build_interpolation_operator(
            model_dict["h"],
            model_dict["xr"],
            model_dict["zr"],
            model_dict["x"],
            model_dict["z"],
            dtype="complex128",
        )

        # Pre-factorize Helmholtz matrices (forward and adjoint)
        print(f"Pre-factorizing {self.nf} Helmholtz matrices...")
        self.A_solvers = []
        self.AH_solvers = []

        for freq in self.f:
            A = build_helmholtz_matrix(freq, self.m, self.h, self.n)

            # Forward solver
            self.A_solvers.append(sparse.linalg.factorized(A.tocsc()))

            # Adjoint solver (conjugate transpose)
            AH = A.conj().T.tocsc()
            self.AH_solvers.append(sparse.linalg.factorized(AH))

    def _matvec(self, dm):
        """
        Forward Jacobian action: J @ dm

        For each frequency:
            1. Compute G·dm (sensitivity kernel times model perturbation)
            2. Solve A·δu = -G·dm for perturbed wavefield
            3. Sample at receivers: δd = Pr·δu
        """
        dm = np.asarray(dm).ravel()

        # Output in (nr, ns, nf) order to match solve() and _rmatvec
        D = np.zeros((self.nr, self.ns, self.nf), dtype=np.complex128)

        for i_freq, freq in enumerate(self.f):
            # Compute RHS for all sources: Rk[:, s] = -G_s @ dm
            Rk = np.zeros((self.N, self.ns), dtype=np.complex128)

            for i_src in range(self.ns):
                u_src = self.U[:, i_src, i_freq]
                G = build_sensitivity_kernel(freq, self.m, u_src, self.h, self.n)
                Rk[:, i_src] = -(G @ dm.reshape(-1, 1)).ravel()

            # Solve A·δu = Rk for all sources at once
            delta_u = self.A_solvers[i_freq](Rk)

            # Sample at receivers: (nr, ns)
            D[:, :, i_freq] = (self.Pr @ delta_u).reshape(self.nr, self.ns)

        return D.reshape(-1, 1)

    def _rmatvec(self, dd):
        """
        Hermitian transpose (adjoint) action: J^H @ dd

        For each frequency:
            1. Reshape dd to (nr, ns, nf)
            2. Back-project to grid: Pr^T·dd
            3. Solve adjoint Helmholtz: A^H·v = Pr^T·dd
            4. Apply adjoint sensitivity: gradient += -G^H·v
        """
        # Reshape data residual to (nr, ns, nf) — matches solve() output order
        dd_reshaped = np.asarray(dd).reshape(self.nr, self.ns, self.nf)

        gradient = np.zeros(self.N, dtype=np.complex128)

        for i_freq, freq in enumerate(self.f):
            # Get data for this frequency: (nr, ns)
            dd_freq = dd_reshaped[:, :, i_freq]

            # Back-project to grid: Pr^T @ dd
            # PyLops Bilinear.T returns 3D (nz, nx, ns), reshape to 2D
            Pr_T_dd = (self.Pr.T @ dd_freq).reshape(self.N, self.ns)

            # Solve adjoint Helmholtz for all sources: A^H·v = Pr^T·dd
            adjoint_wavefield = self.AH_solvers[i_freq](Pr_T_dd)

            # Ensure 2D shape
            if adjoint_wavefield.ndim == 1:
                adjoint_wavefield = adjoint_wavefield.reshape(-1, 1)

            # Apply adjoint sensitivity for each source
            for i_src in range(self.ns):
                u_src = self.U[:, i_src, i_freq]
                G = build_sensitivity_kernel(freq, self.m, u_src, self.h, self.n)

                # G^H @ v = conj(G) @ v for diagonal G
                v_src = adjoint_wavefield[:, i_src]
                result = G.conj() @ v_src

                # Convert sparse result to array if needed
                if hasattr(result, "toarray"):
                    gradient -= result.toarray().squeeze()
                else:
                    gradient -= result

        return gradient.reshape(-1, 1)


class ForwardModeler:
    """
    Forward solver for frequency-domain FWI.

    Solves Helmholtz equation: A(ω,m)·u = q
    Maps model m to data d: d = Pr·u

    Parameters
    ----------
    model_dict : dict
        Dictionary containing:
        - 'h': [dz, dx] grid spacing
        - 'n': [nz, nx] grid dimensions
        - 'f': frequencies (Hz)
        - 'xs', 'zs': source positions
        - 'xr', 'zr': receiver positions
        - 'q': source wavelets
        - 'x', 'z': coordinate vectors
    """

    def __init__(self, model_dict):
        self.model = model_dict

        # Extract dimensions
        self.nz, self.nx = model_dict["n"]
        self.N = self.nz * self.nx
        self.ns = len(model_dict["zs"])
        self.nr = len(model_dict["zr"])
        self.nf = len(model_dict["f"])

        # Build interpolation operators
        self.Ps = build_interpolation_operator(
            model_dict["h"],
            model_dict["xs"],
            model_dict["zs"],
            model_dict["x"],
            model_dict["z"],
            dtype="complex128",
        )

        self.Pr = build_interpolation_operator(
            model_dict["h"],
            model_dict["xr"],
            model_dict["zr"],
            model_dict["x"],
            model_dict["z"],
            dtype="complex128",
        )

        # Build source term: Q = Ps^T @ q
        # PyLops Bilinear.T may return 3D (nz, nx, ns), reshape to 2D (N, ns)
        self.Q = (self.Ps.T @ model_dict["q"]).reshape(self.N, self.ns)

    def solve(self, model_params, return_wavefields=False):
        """
        Solve forward problem for all frequencies and sources.

        Parameters
        ----------
        model_params : ndarray, shape (N, 1) or (N,)
            Slowness-squared model
        return_wavefields : bool
            If True, return full wavefields instead of sampled data

        Returns
        -------
        data : ndarray, shape (nr*ns*nf, 1)
            Simulated data at receivers
        jacobian : JacobianOperator
            Jacobian operator for gradient computation
        """
        m = np.asarray(model_params).ravel()

        # Allocate storage
        U = np.zeros((self.N, self.ns, self.nf), dtype=np.complex128)
        D = np.zeros((self.nr, self.ns, self.nf), dtype=np.complex128)

        print(f"Solving forward problem: {self.nf} frequencies, {self.ns} sources...")

        for i_freq, freq in enumerate(self.model["f"]):
            # Build and factorize Helmholtz matrix
            A = build_helmholtz_matrix(freq, m, self.model["h"], self.model["n"])
            A_solver = sparse.linalg.factorized(A.tocsc())

            # Solve for all sources: A·U = Q
            # factorized() can handle 2D RHS directly
            U[:, :, i_freq] = A_solver(self.Q)

            # Sample at receivers
            D_freq = self.Pr @ U[:, :, i_freq]

            # Handle potential 3D output
            if D_freq.ndim == 3:
                D[:, :, i_freq] = D_freq.reshape(self.nr, self.ns)
            else:
                D[:, :, i_freq] = D_freq.reshape(self.nr, self.ns)

        # Build Jacobian operator
        jacobian = JacobianOperator(m, U, self.model)

        if return_wavefields:
            return U, jacobian
        else:
            return D.reshape(-1, 1), jacobian


# Backward compatibility alias
ForwardSolver = ForwardModeler
