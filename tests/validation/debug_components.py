"""
Deep debugging of the Jacobian operator to find where adjoint breaks.
"""

import numpy as np
from simplefwi.core_v2 import ForwardModeler, build_sensitivity_kernel, build_helmholtz_matrix
from scipy import sparse

# Very simple test
n = [5, 5]
h = [10.0, 10.0]
N = 25

z = np.arange(5) * 10.0
x = np.arange(5) * 10.0

model = {
    "h": h,
    "n": n,
    "f": np.array([2.0]),
    "xs": np.array([20.0]),
    "zs": np.array([10.0]),
    "xr": np.array([10.0, 30.0]),
    "zr": np.array([10.0, 10.0]),
    "q": np.eye(1),
    "x": x,
    "z": z,
}

v = 2.0 * np.ones((5, 5))
m = (1.0 / v**2).reshape(-1, 1)

print("Setting up...")
Fm = ForwardModeler(model)
D, J = Fm.solve(m)

print(f"Data shape: {D.shape}")
print(f"J shape: {J.shape}")

# Get access to internal components
freq = model["f"][0]
U = J.U  # Forward wavefields (N, ns, nf)
u_src = U[:, 0, 0]  # First source, first frequency

print(f"\nWavefield shape: {U.shape}")
print(f"u_src shape: {u_src.shape}")

# Build sensitivity kernel
G = build_sensitivity_kernel(freq, m, u_src, h, n)
print(f"G shape: {G.shape}")
print(f"G is diagonal? {(G - sparse.diags(G.diagonal())).nnz == 0}")

# Build Helmholtz
A = build_helmholtz_matrix(freq, m, h, n)
print(f"A shape: {A.shape}")

# Test individual components
print("\n" + "=" * 60)
print("Testing individual adjoint components")
print("=" * 60)

# Test 1: G adjoint
print("\n1. Testing G adjoint (diagonal):")
dm_test = np.random.randn(N) + 1j * np.random.randn(N)
v_test = np.random.randn(N) + 1j * np.random.randn(N)

Gdm = G @ dm_test
GH_v = G.conj() @ v_test

lhs1 = np.vdot(v_test, Gdm)
rhs1 = np.vdot(GH_v, dm_test)
print(f"  <v, G@dm> = {lhs1}")
print(f"  <G^H@v, dm> = {rhs1}")
print(f"  Error: {abs(lhs1 - rhs1) / abs(lhs1) * 100:.2e}%")

# Test 2: A^{-1} adjoint
print("\n2. Testing A^{-1} adjoint:")
A_solver = sparse.linalg.factorized(A.tocsc())
AH_solver = sparse.linalg.factorized(A.conj().T.tocsc())

rhs_test = np.random.randn(N) + 1j * np.random.randn(N)
y_test = np.random.randn(N) + 1j * np.random.randn(N)

Ainv_rhs = A_solver(rhs_test)
AHinv_y = AH_solver(y_test)

lhs2 = np.vdot(y_test, Ainv_rhs)
rhs2 = np.vdot(AHinv_y, rhs_test)
print(f"  <y, A^{{-1}}@rhs> = {lhs2}")
print(f"  <A^{{-H}}@y, rhs> = {rhs2}")
print(f"  Error: {abs(lhs2 - rhs2) / abs(lhs2) * 100:.2e}%")

# Test 3: Pr adjoint
print("\n3. Testing Pr adjoint:")
u_grid = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
d_recv = np.random.randn(2, 1) + 1j * np.random.randn(2, 1)

Pr_u = J.Pr @ u_grid
PrT_d = J.Pr.T @ d_recv

# Handle PyLops 3D output
if PrT_d.ndim == 3:
    PrT_d = PrT_d.reshape(N, 1)

lhs3 = np.vdot(d_recv.ravel(), Pr_u.ravel())
rhs3 = np.vdot(PrT_d.ravel(), u_grid.ravel())
print(f"  <d, Pr@u> = {lhs3}")
print(f"  <Pr^H@d, u> = {rhs3}")
print(f"  Error: {abs(lhs3 - rhs3) / abs(lhs3) * 100:.2e}%")

# Test 4: Full chain A^{-1} @ (-G @ dm)
print("\n4. Testing full forward chain:")
dm_full = np.random.randn(N) + 1j * np.random.randn(N)
dd_full = np.random.randn(2) + 1j * np.random.randn(2)

# Forward: Pr @ A^{-1} @ (-G @ dm)
Gdm_full = -(G @ dm_full)
Ainv_Gdm = A_solver(Gdm_full)
Pr_Ainv_Gdm = (J.Pr @ Ainv_Gdm.reshape(-1, 1)).ravel()

# Adjoint: -G^H @ A^{-H} @ Pr^H @ dd
PrT_dd = J.Pr.T @ dd_full.reshape(-1, 1)
if PrT_dd.ndim == 3:
    PrT_dd = PrT_dd.reshape(N, 1)
AHinv_PrT = AH_solver(PrT_dd.ravel())
GH_AHinv = -(G.conj() @ AHinv_PrT)

lhs4 = np.vdot(dd_full, Pr_Ainv_Gdm)
rhs4 = np.vdot(GH_AHinv, dm_full)
print(f"  <dd, chain@dm> = {lhs4}")
print(f"  <adjoint@dd, dm> = {rhs4}")
print(f"  Error: {abs(lhs4 - rhs4) / abs(lhs4) * 100:.2e}%")

print("\n" + "=" * 60)
print("Component-wise adjoint verification complete")
print("=" * 60)
