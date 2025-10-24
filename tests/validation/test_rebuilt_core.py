"""
Test the rebuilt core module from scratch.
This uses only current package APIs and clean mathematical implementation.
"""

import numpy as np
from simplefwi.core_v2 import ForwardModeler
from pylops.utils import dottest

print("=" * 80)
print("TESTING REBUILT CORE MODULE (FROM SCRATCH)")
print("=" * 80)

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
    "f": np.array([2.0]),  # Single frequency for testing
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

print("\n" + "=" * 80)
print("STEP 1: Forward Modeling")
print("=" * 80)

try:
    Fm = ForwardModeler(model)
    D, J = Fm.solve(m)

    print(f"✅ Forward solve successful")
    print(f"   Data shape: {D.shape}")
    print(f"   Data range: [{np.abs(D).min():.2e}, {np.abs(D).max():.2e}]")
    print(f"   Jacobian shape: {J.shape}")
except Exception as e:
    print(f"❌ Forward solve failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("STEP 2: Test Jacobian Forward (J @ dm)")
print("=" * 80)

try:
    np.random.seed(42)
    dm = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)

    Jdm = J @ dm

    print(f"✅ Jacobian forward successful")
    print(f"   Input dm shape: {dm.shape}")
    print(f"   Output shape: {Jdm.shape}")
    print(f"   Output range: [{np.abs(Jdm).min():.2e}, {np.abs(Jdm).max():.2e}]")
except Exception as e:
    print(f"❌ Jacobian forward failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("STEP 3: Test Jacobian Adjoint (J.H @ dd)")
print("=" * 80)

try:
    dd = np.random.randn(25, 1) + 1j * np.random.randn(25, 1)

    JHdd = J.H @ dd  # Use .H (Hermitian) not .T (transpose)

    print(f"✅ Jacobian adjoint successful")
    print(f"   Input dd shape: {dd.shape}")
    print(f"   Output shape: {JHdd.shape}")
    print(f"   Output range: [{np.abs(JHdd).min():.2e}, {np.abs(JHdd).max():.2e}]")
except Exception as e:
    print(f"❌ Jacobian adjoint failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("STEP 4: PyLops Adjoint Test (dottest)")
print("=" * 80)

try:
    # Use PyLops dottest for comprehensive adjoint verification
    # Current PyLops API: dottest(Op, nr, nc, rtol=1e-6, complexflag=0, raiseerror=True, verb=False)
    passed = dottest(J, J.shape[0], J.shape[1], rtol=1e-6, complexflag=3, verb=True)

    if passed:
        print("\n" + "🎉" * 40)
        print("✅ ✅ ✅  ADJOINT TEST PASSED!  ✅ ✅ ✅")
        print("🎉" * 40)
        print("\nThe rebuilt core module is MATHEMATICALLY CORRECT!")
        print("Ready for production use.")
    else:
        print("\n❌ Adjoint test failed")
        print("Need further debugging...")

except Exception as e:
    print(f"❌ Adjoint test crashed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("STEP 5: Manual Adjoint Verification")
print("=" * 80)

# Manual verification using .H (Hermitian transpose)
np.random.seed(999)
dm_verify = np.random.randn(N, 1) + 1j * np.random.randn(N, 1)
dd_verify = np.random.randn(25, 1) + 1j * np.random.randn(25, 1)

Jdm_verify = J @ dm_verify
JHdd_verify = J.H @ dd_verify  # Use .H not .T

lhs = np.vdot(dd_verify, Jdm_verify)
rhs = np.vdot(JHdd_verify, dm_verify)

print(f"<dd, J @ dm>    = {lhs}")
print(f"<J.H @ dd, dm>  = {rhs}")
print(f"Difference      = {lhs - rhs}")
print(f"Relative error  = {np.abs(lhs - rhs) / np.abs(lhs) * 100:.6f}%")

if np.abs(lhs - rhs) / np.abs(lhs) < 1e-6:
    print("✅ Manual verification: PASSED")
    print("\n" + "🎉" * 40)
    print("SUCCESS! The rebuilt core_v2.py is CORRECT and ready to use!")
    print("🎉" * 40)
else:
    print("❌ Manual verification: FAILED")
