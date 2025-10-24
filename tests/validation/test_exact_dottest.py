"""
Test the actual _matvec and _rmatvec with the exact same test vectors PyLops uses.
"""

import numpy as np
from simplefwi.core_v2 import ForwardModeler

# Simple test matching PyLops dottest
n = [10, 10]
h = [10.0, 10.0]
N = 100

z = np.arange(10) * 10.0
x = np.arange(10) * 10.0

model = {
    "h": h,
    "n": n,
    "f": np.array([2.0]),
    "xs": np.array([50.0]),
    "zs": np.array([10.0]),
    "xr": np.array([20.0, 40.0, 60.0]),
    "zr": np.array([10.0, 10.0, 10.0]),
    "q": np.eye(1),
    "x": x,
    "z": z,
}

v = 2.0 * np.ones((10, 10))
m = (1.0 / v**2).reshape(-1, 1)

Fm = ForwardModeler(model)
D, J = Fm.solve(m)

print(f"J shape: {J.shape}")  # (3, 100)
print(f"nr={J.nr}, ns={J.ns}, nf={J.nf}")

# Create test vectors EXACTLY like PyLops dottest does
np.random.seed(0)

# For complexflag=3, PyLops creates:
# u = randn(nc) + 1j*randn(nc)  where nc = J.shape[1] = 100
# v = randn(nr) + 1j*randn(nr)  where nr = J.shape[0] = 3

u = np.random.randn(J.shape[1]) + 1j * np.random.randn(J.shape[1])  # (100,)
v = np.random.randn(J.shape[0]) + 1j * np.random.randn(J.shape[0])  # (3,)

print(f"\nTest vectors:")
print(f"u shape: {u.shape}, dtype: {u.dtype}")
print(f"v shape: {v.shape}, dtype: {v.dtype}")

# PyLops will reshape to column vectors
u_col = u.reshape(-1, 1)
v_col = v.reshape(-1, 1)

# Forward
print("\n" + "=" * 60)
print("Testing _matvec")
print("=" * 60)
y = J._matvec(u_col)
print(f"Input shape: {u_col.shape}")
print(f"Output shape: {y.shape}")
print(f"Output range: [{np.abs(y).min():.2e}, {np.abs(y).max():.2e}]")

# Adjoint
print("\n" + "=" * 60)
print("Testing _rmatvec")
print("=" * 60)
x = J._rmatvec(v_col)
print(f"Input shape: {v_col.shape}")
print(f"Output shape: {x.shape}")
print(f"Output range: [{np.abs(x).min():.2e}, {np.abs(x).max():.2e}]")

# Adjoint test - exactly as PyLops does it
print("\n" + "=" * 60)
print("Adjoint test (PyLops style)")
print("=" * 60)

# PyLops computes: v^H @ (Op @ u) vs u^H @ (Op^H @ v)
yy = np.vdot(v_col.ravel(), y.ravel())
xx = np.vdot(x.ravel(), u_col.ravel())

print(f"v^H @ (J @ u)    = {yy}")
print(f"u^H @ (J^H @ v)  = {xx}")
print(f"Difference       = {yy - xx}")
print(f"Relative error   = {np.abs(yy - xx) / np.abs(yy) * 100:.6f}%")

if np.abs(yy - xx) / np.abs(yy) < 1e-6:
    print("\n✅ PASSES adjoint test!")
else:
    print("\n❌ FAILS adjoint test")

    # More debugging
    print("\nDetailed analysis:")
    print(f"  |yy| = {np.abs(yy):.6e}")
    print(f"  |xx| = {np.abs(xx):.6e}")
    print(f"  yy.real = {yy.real:.6e}, yy.imag = {yy.imag:.6e}")
    print(f"  xx.real = {xx.real:.6e}, xx.imag = {xx.imag:.6e}")

    # Check if signs are flipped
    if np.allclose(yy, -xx):
        print("\n  -> Values are negatives of each other! Sign error in adjoint.")
    if np.allclose(yy, np.conj(xx)):
        print("\n  -> Values are conjugates! Missing conjugate in adjoint.")
    if np.allclose(yy, -np.conj(xx)):
        print("\n  -> Values are negative conjugates! Sign + conjugate error.")
