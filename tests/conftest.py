"""
Shared pytest fixtures and configuration for SimpleFWI tests.
"""

import pytest
import numpy as np


@pytest.fixture
def small_model():
    """
    Small test model (50×50 grid).

    Returns
    -------
    dict
        Model dictionary with all necessary parameters.
    """
    nz, nx = 50, 50
    dz, dx = 10.0, 10.0

    # Grid coordinates
    z = np.arange(nz) * dz
    x = np.arange(nx) * dx

    model = {
        "h": [dz, dx],
        "n": [nz, nx],
        "f": [2.0, 3.0],  # Two frequencies
        "xs": [250.0],  # Single source at center
        "zs": [10.0],
        "xr": np.arange(0, nx * dx, 50.0),  # 10 receivers
        "zr": 10.0 * np.ones(10),
        "q": np.eye(1),  # Identity source wavelet
        "x": x,
        "z": z,
    }

    return model


@pytest.fixture
def medium_model():
    """
    Medium test model (100×100 grid).

    Returns
    -------
    dict
        Model dictionary with all necessary parameters.
    """
    nz, nx = 100, 100
    dz, dx = 10.0, 10.0

    z = np.arange(nz) * dz
    x = np.arange(nx) * dx

    model = {
        "h": [dz, dx],
        "n": [nz, nx],
        "f": [2.0, 3.0, 4.0],  # Three frequencies
        "xs": [500.0],
        "zs": [10.0],
        "xr": np.arange(0, nx * dx, 20.0),  # 50 receivers
        "zr": 10.0 * np.ones(50),
        "q": np.eye(1),
        "x": x,
        "z": z,
    }

    return model


@pytest.fixture
def homogeneous_slowness_squared():
    """
    Homogeneous slowness-squared model (m = 1/v²).

    Returns
    -------
    callable
        Function that takes n (grid dimensions) and returns m (N, 1) array.
    """

    def _make_model(n, velocity=2.0):
        """
        Create homogeneous model.

        Parameters
        ----------
        n : list
            Grid dimensions [nz, nx]
        velocity : float, optional
            Constant velocity in km/s (default: 2.0)

        Returns
        -------
        m : ndarray, shape (N, 1)
            Slowness-squared model
        """
        N = int(np.prod(n))
        m = (1.0 / velocity) ** 2 * np.ones((N, 1))
        return m

    return _make_model


@pytest.fixture
def disk_anomaly_model():
    """
    Circular disk anomaly in homogeneous background.

    Returns
    -------
    callable
        Function that takes model dict and returns (m_true, m0).
    """

    def _make_model(model, v_background=2.0, v_anomaly=3.0, radius_fraction=0.2):
        """
        Create disk anomaly model.

        Parameters
        ----------
        model : dict
            Model dictionary with 'n', 'h' keys
        v_background : float
            Background velocity in km/s
        v_anomaly : float
            Anomaly velocity in km/s
        radius_fraction : float
            Radius as fraction of domain size

        Returns
        -------
        m_true : ndarray, shape (N, 1)
            True slowness-squared model
        m0 : ndarray, shape (N, 1)
            Initial (homogeneous) model
        """
        from skimage.draw import disk

        nz, nx = model["n"]
        N = nz * nx

        # Create velocity model
        vel = v_background * np.ones((nz, nx))

        # Add circular anomaly
        center = (nz // 2, nx // 2)
        radius = int(radius_fraction * min(nz, nx))
        rr, cc = disk(center, radius, shape=(nz, nx))
        vel[rr, cc] = v_anomaly

        # Convert to slowness-squared
        m_true = (1.0 / vel.ravel()).reshape(-1, 1) ** 2
        m0 = (1.0 / v_background) ** 2 * np.ones((N, 1))

        return m_true, m0

    return _make_model


@pytest.fixture
def backend_available():
    """
    Check which backends are available.

    Returns
    -------
    dict
        Dictionary with backend availability.
    """
    backends = {
        "numpy": True,  # Always available
        "cupy": False,
        "jax": False,
        "metal": False,
    }

    # Check CuPy (NVIDIA CUDA)
    try:
        import cupy as cp

        _ = cp.array([1.0])
        backends["cupy"] = True
    except:
        pass

    # Check JAX
    try:
        import jax

        _ = jax.numpy.array([1.0])
        backends["jax"] = True

        # Check Metal specifically (Apple Silicon)
        if "metal" in str(jax.devices()[0]).lower():
            backends["metal"] = True
    except:
        pass

    return backends


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "metal: marks tests for Apple Silicon Metal backend")
    config.addinivalue_line("markers", "cuda: marks tests for NVIDIA CUDA backend")
