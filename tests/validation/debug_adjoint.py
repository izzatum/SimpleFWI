"""
Debug the adjoint implementation step by step.
"""

import numpy as np
from simplefwi.core_v2 import ForwardModeler, build_sensitivity_kernel

# Simple test
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

print("Setting up forward solver...")
Fm = ForwardModeler(model)
D, J = Fm.solve(m)

print(f"Data shape: {D.shape}")
print(f"Jacobian shape: {J.shape}")

# Test with simple vectors
np.random.seed(42)
dm = np.random.randn(N, 1)  # Real perturbation
dd = np.random.randn(3, 1)  # Real data residual

print("\n" + "=" * 60)
print("Testing with REAL vectors first")
print("=" * 60)

# Forward
Jdm = J @ dm
print(f"J @ dm: shape={Jdm.shape}, is_complex={np.iscomplexobj(Jdm)}")

# Adjoint
JTdd = J.T @ dd
print(f"J.T @ dd: shape={JTdd.shape}, is_complex={np.iscomplexobj(JTdd)}")

# Dot test
lhs = np.dot(dd.ravel(), Jdm.ravel())  # Real dot product
rhs = np.dot(JTdd.ravel(), dm.ravel())

print(f"\n<dd, J @ dm>   = {lhs}")
print(f"<J.T @ dd, dm> = {rhs}")
print(f"Difference     = {lhs - rhs}")
print(f"Relative error = {abs(lhs - rhs) / abs(lhs) * 100:.2f}%")

print("\n" + "=" * 60)
print("Testing with COMPLEX vectors")
print("=" * 60)

# Complex test vectors
dm_c = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
dd_c = np.random.randn(3, 1) + 1j * np.random.randn(3, 1)

# Forward
Jdm_c = J @ dm_c
print(f"J @ dm: shape={Jdm_c.shape}")

# Adjoint
JTdd_c = J.T @ dd_c
print(f"J.T @ dd: shape={JTdd_c.shape}")

# Complex dot test: <v, w> = v^H @ w = conj(v) · w
lhs_c = np.vdot(dd_c.ravel(), Jdm_c.ravel())
rhs_c = np.vdot(JTdd_c.ravel(), dm_c.ravel())

print(f"\n<dd, J @ dm>   = {lhs_c}")
print(f"<J.T @ dd, dm> = {rhs_c}")
print(f"Difference     = {lhs_c - rhs_c}")
print(f"Relative error = {abs(lhs_c - rhs_c) / abs(lhs_c) * 100:.2f}%")

if abs(lhs_c - rhs_c) / abs(lhs_c) < 0.01:
    print("\n✅ Adjoint is reasonably correct (< 1% error)")
else:
    print("\n❌ Adjoint has significant error")

    # Debug: check individual components
    print("\nDEBUGGING:")
    print(f"lhs real: {lhs_c.real:.6e}, imag: {lhs_c.imag:.6e}")
    print(f"rhs real: {rhs_c.real:.6e}, imag: {rhs_c.imag:.6e}")
