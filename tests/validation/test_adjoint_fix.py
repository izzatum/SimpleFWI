"""
Test adjoint implementation after fix.
If this fails, we'll rebuild from scratch.
"""

import numpy as np
from simplefwi.core import ForwardSolver

# Test model parameters
n = [50, 50]
h = [10.0, 10.0]
nz, nx = n
N = nz * nx

# Coordinate vectors
z = np.arange(nz) * h[0]
x = np.arange(nx) * h[1]

# Source and receiver positions (AWAY FROM EDGES)
xs = np.array([250.0])
zs = np.array([10.0])
xr = np.linspace(10, 480, 25)
zr = np.ones_like(xr) * 10.0

# Model dictionary
model = {
    "h": h,
    "n": n,
    "f": np.array([2.0]),  # Single frequency
    "xs": xs,
    "zs": zs,
    "xr": xr,
    "zr": zr,
    "q": np.eye(1),  # Identity source
    "x": x,
    "z": z,
}

# Create simple velocity model
v = 2.0 * np.ones((nz, nx))  # 2 km/s background
m = (1.0 / v**2).reshape(-1, 1)  # Slowness-squared

print("=" * 70)
print("COMPREHENSIVE ADJOINT TEST")
print("=" * 70)

# Forward solve
print("\n1. Forward modeling...")
Fm = ForwardSolver(model)
D, DF = Fm.solve(m)
print(f"   Data shape: {D.shape}")
print(f"   Data range: [{np.abs(D).min():.1f}, {np.abs(D).max():.1f}]")

# Disable Numba to test NumPy path
print("\n2. Testing NumPy path (Numba disabled)...")
DF.use_numba = False

# Generate random test vectors
np.random.seed(42)
dm = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
dd = np.random.randn(25, 1) + 1j * np.random.randn(25, 1)

# Test Jacobian forward
print("\n3. Testing Jacobian forward (J @ dm)...")
Jdm = DF @ dm
print(f"   Result shape: {Jdm.shape}")
print(f"   Result range: [{np.abs(Jdm).min():.2e}, {np.abs(Jdm).max():.2e}]")

# Test Jacobian adjoint
print("\n4. Testing Jacobian adjoint (J.T @ dd)...")
JTdd = DF.T @ dd
print(f"   Result shape: {JTdd.shape}")
print(f"   Result range: [{np.abs(JTdd).min():.2e}, {np.abs(JTdd).max():.2e}]")

# Adjoint test
print("\n5. Adjoint test: <dd, J @ dm> should equal <J.T @ dd, dm>")
lhs = np.vdot(dd, Jdm)
rhs = np.vdot(JTdd, dm)

print(f"   <dd, J @ dm>    = {lhs}")
print(f"   <J.T @ dd, dm>  = {rhs}")
print(f"   Difference      = {lhs - rhs}")
print(f"   Relative error  = {np.abs(lhs - rhs) / np.abs(lhs) * 100:.2f}%")

if np.abs(lhs - rhs) / np.abs(lhs) < 1e-6:
    print("\n" + "=" * 70)
    print("✅ ADJOINT TEST PASSED! NumPy path is CORRECT.")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("❌ ADJOINT TEST FAILED! Need to rebuild from scratch.")
    print("=" * 70)
    print("\nDEBUGGING INFO:")
    print(f"lhs magnitude: {np.abs(lhs):.6e}")
    print(f"rhs magnitude: {np.abs(rhs):.6e}")
    print(f"lhs real: {lhs.real:.6e}, imag: {lhs.imag:.6e}")
    print(f"rhs real: {rhs.real:.6e}, imag: {rhs.imag:.6e}")
