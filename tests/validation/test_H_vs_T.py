"""
Investigate why PyLops dottest passes but manual test fails.
Let's reproduce EXACTLY what PyLops does internally.
"""

import numpy as np
from simplefwi.core_v2 import ForwardModeler

# Same setup
n = [50, 50]
h = [10.0, 10.0]
nz, nx = n
N = nz * nx

z = np.arange(nz) * h[0]
x = np.arange(nx) * h[1]

xs = np.array([250.0])
zs = np.array([10.0])
xr = np.linspace(10, 480, 25)
zr = np.ones_like(xr) * 10.0

model = {
    "h": h,
    "n": n,
    "f": np.array([2.0]),
    "xs": xs,
    "zs": zs,
    "xr": xr,
    "zr": zr,
    "q": np.eye(1),
    "x": x,
    "z": z,
}

v = 2.0 * np.ones((nz, nx))
m = (1.0 / v**2).reshape(-1, 1)

Fm = ForwardModeler(model)
D, J = Fm.solve(m)

print(f"Operator shape: {J.shape}")
print(f"nr={J.nr}, ns={J.ns}, nf={J.nf}")
print(f"Expected output size: nr*ns*nf = {J.nr*J.ns*J.nf}")

# Look at PyLops source code for dottest with complexflag=3:
# It creates: u = randn(nc) + 1j*randn(nc) and v = randn(nr) + 1j*randn(nr)
# Then tests: vdot(v, Op @ u) vs vdot(Op.H @ v, u)

np.random.seed(12345)
u = np.random.randn(J.shape[1]) + 1j * np.random.randn(J.shape[1])
v = np.random.randn(J.shape[0]) + 1j * np.random.randn(J.shape[0])

print(f"\nTest vector shapes:")
print(f"u: {u.shape} (should be {J.shape[1]})")
print(f"v: {v.shape} (should be {J.shape[0]})")

# Reshape to column vectors (PyLops expects this)
u = u.reshape(-1, 1)
v = v.reshape(-1, 1)

print(f"\nAfter reshape:")
print(f"u: {u.shape}")
print(f"v: {v.shape}")

# Test using @ operator (which calls matvec/rmatvec)
print("\n" + "=" * 70)
print("Using @ operator (what PyLops does)")
print("=" * 70)

y = J @ u
x = J.H @ v  # Note: .H calls _rmatvec

print(f"J @ u shape: {y.shape}")
print(f"J.H @ v shape: {x.shape}")

lhs_op = np.vdot(v, y)
rhs_op = np.vdot(x, u)

print(f"\nvdot(v, J @ u)    = {lhs_op}")
print(f"vdot(J.H @ v, u)  = {rhs_op}")
print(f"Difference        = {lhs_op - rhs_op}")
print(f"Relative error    = {abs(lhs_op - rhs_op) / abs(lhs_op) * 100:.6f}%")

# Now test using .T operator
print("\n" + "=" * 70)
print("Using .T operator (what we're doing in manual test)")
print("=" * 70)

y2 = J @ u
x2 = J.T @ v  # Note: .T also calls _rmatvec

print(f"J @ u shape: {y2.shape}")
print(f"J.T @ v shape: {x2.shape}")

lhs_t = np.vdot(v, y2)
rhs_t = np.vdot(x2, u)

print(f"\nvdot(v, J @ u)   = {lhs_t}")
print(f"vdot(J.T @ v, u) = {rhs_t}")
print(f"Difference       = {lhs_t - rhs_t}")
print(f"Relative error   = {abs(lhs_t - rhs_t) / abs(lhs_t) * 100:.6f}%")

# Check if .H and .T give same result
print("\n" + "=" * 70)
print("Comparing .H vs .T")
print("=" * 70)
print(f"J.H @ v equals J.T @ v? {np.allclose(x, x2)}")
print(f"Max difference: {np.max(np.abs(x - x2))}")

if not np.allclose(x, x2):
    print("\n❌ WARNING: .H and .T give different results!")
    print("This means _rmatvec might not implement true Hermitian transpose")
