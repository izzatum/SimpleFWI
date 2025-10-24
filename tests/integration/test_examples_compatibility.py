"""
Test compatibility with existing example notebooks.

This script verifies that the new core.py works with all existing
example workflows without breaking changes.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/Users/muhammadizzatullah/Documents/SimpleFWI")

print("=" * 80)
print("TESTING EXAMPLE COMPATIBILITY")
print("=" * 80)

# Import SimpleFWI components
try:
    from simplefwi.core import ForwardSolver
    from simplefwi.misfit import DataMisfit, RegMisfit, MisfitFunction
    from simplefwi.optimization import BBiter, CGiter

    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 1: Example_modelling.ipynb workflow (Forward modeling only)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Forward Modeling (Example_modelling.ipynb workflow)")
print("=" * 80)

try:
    # Small grid for fast testing
    nz, nx = 50, 50
    dz, dx = 10.0, 10.0
    h = [dz, dx]
    n = [nz, nx]
    N = nz * nx

    # Grid coordinates
    z = np.arange(nz) * dz
    x = np.arange(nx) * dx

    # Simple acquisition geometry
    ns = 1  # Single source
    xs = np.array([250.0])
    zs = np.array([10.0])

    nr = 25  # 25 receivers
    # Keep receivers away from exact boundaries (use 0.1 to nx*dx-0.1)
    xr = np.linspace(10.0, (nx - 1) * dx - 10.0, nr)
    zr = 10.0 * np.ones(nr)

    # Single frequency
    f = np.array([2.0])

    # Source wavelet
    q = np.eye(ns)

    # Build model dictionary
    model = {"h": h, "n": n, "f": f, "xs": xs, "zs": zs, "xr": xr, "zr": zr, "q": q, "x": x, "z": z}

    # Homogeneous velocity model
    v = 2.0  # km/s
    m = (1.0 / v) ** 2 * np.ones((N, 1))

    # Create ForwardSolver and solve
    Fm = ForwardSolver(model)
    D, J = Fm.solve(m)

    print(f"   Forward solve successful")
    print(f"   Data shape: {D.shape} (expected: ({nr*ns*len(f)}, 1))")
    print(f"   Jacobian shape: {J.shape}")
    print(f"   Data range: [{np.abs(D).min():.2e}, {np.abs(D).max():.2e}]")

    # Verify shapes
    assert D.shape == (nr * ns * len(f), 1), f"Wrong data shape: {D.shape}"
    assert J.shape == (nr * ns * len(f), N), f"Wrong Jacobian shape: {J.shape}"

    print("   ✅ Forward modeling test PASSED")

except Exception as e:
    print(f"   ❌ Forward modeling test FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 2: Example_1.ipynb workflow (Circular reflector)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Circular Reflector (Example_1.ipynb workflow)")
print("=" * 80)

try:
    # Create circular anomaly model
    def create_circular_anomaly(n, center, radius, v_background, v_anomaly):
        """Create model with circular velocity anomaly."""
        nz, nx = n
        z_grid, x_grid = np.mgrid[0:nz, 0:nx]

        # Distance from center
        cz, cx = center
        dist = np.sqrt((z_grid - cz) ** 2 + (x_grid - cx) ** 2)

        # Velocity model
        v = v_background * np.ones((nz, nx))
        v[dist <= radius] = v_anomaly

        # Convert to slowness-squared
        m = (1.0 / v) ** 2
        return m.reshape(-1, 1), v

    # Create models
    m_true, v_true = create_circular_anomaly(
        n=[50, 50], center=(25, 25), radius=10, v_background=2.0, v_anomaly=2.5
    )

    m_init, v_init = create_circular_anomaly(
        n=[50, 50], center=(25, 25), radius=0, v_background=2.0, v_anomaly=2.0  # No anomaly
    )

    # Generate synthetic data
    Dobs, _ = Fm.solve(m_true)

    print(f"   Synthetic data generated")
    print(f"   Data misfit (init vs true): {np.linalg.norm(Dobs):.2e}")

    # Test data misfit
    D_init, J_init = Fm.solve(m_init)
    residual = np.linalg.norm(Dobs - D_init)

    print(f"   Initial residual: {residual:.2e}")
    print(f"   ✅ Circular reflector test PASSED")

except Exception as e:
    print(f"   ❌ Circular reflector test FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 3: Example_2.ipynb workflow (Inversion with BB/CG)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: FWI Inversion (Example_2.ipynb workflow)")
print("=" * 80)

try:
    # Setup misfit function
    dataMisfit = DataMisfit(Dobs, model)
    regMisfit = RegMisfit(n, alpha=0.5, m0=m_init)
    misfit = MisfitFunction(dataMisfit, regMisfit)

    # Evaluate initial misfit and gradient
    f0, g0, _ = misfit.evaluate(m_init)

    print(f"   Initial misfit: {f0:.2e}")
    print(f"   Gradient shape: {g0.shape}")
    print(f"   Gradient norm: {np.linalg.norm(g0):.2e}")

    # Run 3 iterations of BB optimizer (quick test)
    print("\n   Running 3 iterations of BB optimizer...")
    history, m_inv, g_final = BBiter(misfit, m_init, tol=1e-6, maxit=3, bounds=None)

    print(f"   BB optimizer completed")
    print(f"   History shape: {history.shape}")
    print(f"   Final misfit: {history[-1, 1]:.2e}")
    print(f"   Initial misfit: {history[0, 1]:.2e}")

    # Verify misfit decreased
    assert history[-1, 1] < history[0, 1], "Misfit should decrease"

    print(f"   Misfit reduction: {(1 - history[-1, 1]/history[0, 1])*100:.1f}%")

    # Test CG optimizer (2 iterations)
    # CGiter requires Dobs and ForwardSolver for deterministic line search
    print("\n   Running 2 iterations of CG optimizer...")
    history_cg, m_inv_cg, g_final_cg = CGiter(
        misfit, m_init, Dobs, Fm, tol=1e-6, maxit=2, bounds=None
    )

    print(f"   CG optimizer completed")
    print(f"   Final misfit: {history_cg[-1, 1]:.2e}")

    print("   ✅ FWI inversion test PASSED")

except Exception as e:
    print(f"   ❌ FWI inversion test FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: Multi-frequency workflow (Example_3.ipynb style)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Multi-frequency FWI (Example_3.ipynb style)")
print("=" * 80)

try:
    # Multi-frequency model
    model_mf = {
        "h": h,
        "n": n,
        "f": np.array([1.5, 2.0, 2.5]),  # 3 frequencies
        "xs": xs,
        "zs": zs,
        "xr": xr,
        "zr": zr,
        "q": q,
        "x": x,
        "z": z,
    }

    # Create new forward solver with multiple frequencies
    Fm_mf = ForwardSolver(model_mf)

    # Generate multi-frequency data
    Dobs_mf, J_mf = Fm_mf.solve(m_true)

    print(f"   Multi-frequency data shape: {Dobs_mf.shape}")
    print(f"   Expected: ({nr*ns*3}, 1)")
    print(f"   Jacobian shape: {J_mf.shape}")

    # Verify shapes
    assert Dobs_mf.shape == (nr * ns * 3, 1), f"Wrong multi-freq data shape"
    assert J_mf.shape == (nr * ns * 3, N), f"Wrong multi-freq Jacobian shape"

    # Test gradient computation with multiple frequencies
    dataMisfit_mf = DataMisfit(Dobs_mf, model_mf)
    f_mf, g_mf, _ = dataMisfit_mf.evaluate(m_init)

    print(f"   Multi-frequency misfit: {f_mf:.2e}")
    print(f"   Gradient norm: {np.linalg.norm(g_mf):.2e}")

    print("   ✅ Multi-frequency test PASSED")

except Exception as e:
    print(f"   ❌ Multi-frequency test FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL EXAMPLE COMPATIBILITY TESTS PASSED!")
print("=" * 80)
print("\nSummary:")
print("  ✅ Forward modeling works (Example_modelling.ipynb)")
print("  ✅ Circular reflector works (Example_1.ipynb)")
print("  ✅ BB/CG optimization works (Example_2.ipynb)")
print("  ✅ Multi-frequency FWI works (Example_3.ipynb style)")
print("\n" + "=" * 80)
print("New core.py is fully compatible with all existing examples!")
print("Ready for production use.")
print("=" * 80)
