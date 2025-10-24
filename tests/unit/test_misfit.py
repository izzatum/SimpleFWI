"""
Unit tests for misfit.py functions and classes.

Tests:
- DataMisfit: L2 data misfit computation
- RegMisfit: Tikhonov regularization
- MisfitFunction: Combined objective function
"""

import pytest
import numpy as np


class TestDataMisfit:
    """Test DataMisfit class."""

    def test_data_misfit_evaluation(self, small_model):
        """Test data misfit function value."""
        pytest.skip("To be implemented after moving misfit.py")

    def test_data_misfit_gradient(self, small_model):
        """Test data misfit gradient via adjoint-state."""
        pytest.skip("To be implemented after moving misfit.py")

    def test_gradient_accuracy_finite_difference(self, small_model):
        """
        Test gradient accuracy via finite difference.

        Compare adjoint-state gradient with finite-difference approximation.
        """
        pytest.skip("To be implemented after moving misfit.py")


class TestRegularizationMisfit:
    """Test RegMisfit (Tikhonov regularization)."""

    def test_regularization_value(self, small_model):
        """Test regularization term value."""
        pytest.skip("To be implemented after moving misfit.py")

    def test_regularization_gradient(self, small_model):
        """Test regularization gradient."""
        pytest.skip("To be implemented after moving misfit.py")

    def test_smoothness_constraint(self, small_model):
        """Test that regularization penalizes roughness."""
        pytest.skip("To be implemented after moving misfit.py")


class TestMisfitFunction:
    """Test MisfitFunction (data + regularization)."""

    def test_combined_misfit_value(self, small_model):
        """Test combined objective function value."""
        pytest.skip("To be implemented after moving misfit.py")

    def test_combined_misfit_gradient(self, small_model):
        """Test combined gradient."""
        pytest.skip("To be implemented after moving misfit.py")

    def test_regularization_weight_effect(self, small_model):
        """Test effect of regularization weight α."""
        pytest.skip("To be implemented after moving misfit.py")
