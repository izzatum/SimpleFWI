"""
Unit tests for core.py functions and classes.

Tests:
- getA(): Helmholtz matrix construction
- getP(): Source/receiver interpolation
- getG(): Sensitivity kernel
- ForwardSolver: Forward modeling
- JacobianForwardSolver: Adjoint test
"""

import pytest
import numpy as np


class TestHelmholtzMatrix:
    """Test getA() Helmholtz matrix construction."""

    def test_helmholtz_shape(self, small_model):
        """Test that Helmholtz matrix has correct shape."""
        pytest.skip("To be implemented after moving core.py")

    def test_helmholtz_symmetry(self, small_model):
        """Test that Helmholtz matrix structure is correct."""
        pytest.skip("To be implemented after moving core.py")

    def test_absorbing_boundary_weights(self, small_model):
        """Test that boundary weights are correctly applied."""
        pytest.skip("To be implemented after moving core.py")


class TestSourceReceiverOperator:
    """Test getP() bilinear interpolation operator."""

    def test_interpolation_accuracy(self, small_model):
        """Test interpolation matches expected values."""
        pytest.skip("To be implemented after moving core.py")

    def test_adjoint_property(self, small_model):
        """Test that P.H is true adjoint of P."""
        pytest.skip("To be implemented after moving core.py")


class TestSensitivityKernel:
    """Test getG() sensitivity kernel."""

    def test_sensitivity_kernel_shape(self, small_model):
        """Test that G has correct shape."""
        pytest.skip("To be implemented after moving core.py")


class TestForwardSolver:
    """Test ForwardSolver class."""

    def test_forward_solve_shape(self, small_model, homogeneous_slowness_squared):
        """Test that forward solve produces correct output shape."""
        pytest.skip("To be implemented after moving core.py")

    def test_forward_solve_reciprocity(self, small_model):
        """Test Green's function reciprocity."""
        pytest.skip("To be implemented after moving core.py")

    @pytest.mark.slow
    def test_forward_solve_convergence(self, medium_model):
        """Test convergence with grid refinement."""
        pytest.skip("To be implemented after moving core.py")


class TestJacobianOperator:
    """Test JacobianForwardSolver adjoint test."""

    def test_adjoint_test_passes(self, small_model, homogeneous_slowness_squared):
        """
        CRITICAL TEST: Verify Jacobian adjoint is correct.

        This test MUST pass for FWI to work correctly.
        """
        pytest.skip("To be implemented after moving core.py")

        # Example of what test should do:
        # from pylops.utils import dottest
        # from simplefwi import ForwardSolver
        #
        # m = homogeneous_slowness_squared(small_model['n'])
        # Fm = ForwardSolver(small_model)
        # D, J = Fm.solve(m)
        #
        # nr = len(small_model['xr'])
        # ns = len(small_model['xs'])
        # nf = len(small_model['f'])
        # N = np.prod(small_model['n'])
        #
        # # Adjoint test (MUST pass with tol < 1e-6)
        # passed = dottest(J, nr*ns*nf, N, tol=1e-6, complexflag=3)
        # assert passed, "Adjoint test failed!"

    @pytest.mark.slow
    def test_jacobian_matvec(self, small_model):
        """Test forward Jacobian-vector product."""
        pytest.skip("To be implemented after moving core.py")

    @pytest.mark.slow
    def test_jacobian_rmatvec(self, small_model):
        """Test adjoint Jacobian-vector product."""
        pytest.skip("To be implemented after moving core.py")


@pytest.mark.gpu
class TestBackendConsistency:
    """Test that different backends produce same results."""

    def test_helmholtz_numpy_vs_cupy(self, small_model, backend_available):
        """Test Helmholtz matrix on CPU vs CUDA GPU."""
        if not backend_available["cupy"]:
            pytest.skip("CuPy not available")

        pytest.skip("To be implemented in Phase 1 (GPU acceleration)")

    def test_helmholtz_numpy_vs_jax(self, small_model, backend_available):
        """Test Helmholtz matrix on CPU vs JAX (Metal/CUDA)."""
        if not backend_available["jax"]:
            pytest.skip("JAX not available")

        pytest.skip("To be implemented in Phase 1 (GPU acceleration)")

    @pytest.mark.metal
    def test_forward_solve_metal(self, small_model, backend_available):
        """Test forward solve on Apple Silicon Metal GPU."""
        if not backend_available["metal"]:
            pytest.skip("Metal GPU not available")

        pytest.skip("To be implemented in Phase 1 (GPU acceleration)")
