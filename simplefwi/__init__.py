"""
SimpleFWI: Performance-grade frequency-domain Full Waveform Inversion
======================================================================

SimpleFWI is a high-performance Python package for frequency-domain Full
Waveform Inversion (FWI) of seismic data. It leverages PyLops for matrix-free
operators and supports multiple backends (NumPy, CuPy, JAX) for CPU and GPU
acceleration.

Main Features
-------------
- **Frequency-domain FWI**: Solves Helmholtz equation for wave propagation
- **Matrix-free operators**: Memory-efficient via PyLops LinearOperator
- **Backend-agnostic**: Automatic support for NumPy, CuPy (CUDA), JAX (Metal/CUDA)
- **Iterative optimization**: Barzilai-Borwein, Conjugate Gradient, and more
- **Adjoint-state gradients**: Efficient gradient computation for large-scale problems

Quick Start
-----------
>>> import numpy as np
>>> from simplefwi import ForwardSolver, DataMisfit, BBiter
>>>
>>> # Define model
>>> model = {
...     'h': [10.0, 10.0],          # Grid spacing [dz, dx]
...     'n': [100, 100],            # Grid dimensions [nz, nx]
...     'f': [2.0, 3.0, 4.0],       # Frequencies in Hz
...     'xs': [500],                # Source x-coordinates
...     'zs': [10],                 # Source z-coordinates
...     'xr': np.arange(0, 1000, 20),  # Receiver x-coordinates
...     'zr': 10*np.ones(50),       # Receiver z-coordinates
...     'q': np.eye(1),             # Source wavelet matrix
... }
>>>
>>> # Create forward solver
>>> m_true = (1/2.0)**2 * np.ones((10000, 1))  # Slowness-squared
>>> Fm = ForwardSolver(model)
>>> Dobs, J = Fm.solve(m_true)
>>>
>>> # Run inversion
>>> m0 = (1/2.5)**2 * np.ones((10000, 1))  # Initial model
>>> misfit = DataMisfit(Dobs, model)
>>> history, m_inv, g = BBiter(misfit, m0, tol=1e-3, maxit=10)

Modules
-------
core
    Forward modeling, Helmholtz matrix construction, Jacobian operators
misfit
    Objective functions: data misfit, regularization, combined misfit
optimization
    Gradient-based optimizers: Barzilai-Borwein, Conjugate Gradient

Development
-----------
SimpleFWI follows SOLID principles with emphasis on performance:
- GPU acceleration via CuPy (NVIDIA) or JAX (Apple Silicon Metal)
- JIT compilation via Numba for computational kernels
- Iterative solvers for large-scale problems (N > 1e6)
- Matrix-free operators to minimize memory footprint

See documentation at: https://github.com/izzatum/SimpleFWI
"""

__version__ = "0.1.0"
__author__ = "Muhammad Izzatullah"
__email__ = "izzatum@users.noreply.github.com"

# Import core functionality
from simplefwi.core import (
    build_helmholtz_matrix,
    build_interpolation_operator,
    build_sensitivity_kernel,
    ForwardModeler,
    ForwardSolver,  # Backward compatibility alias
    JacobianOperator,
)

from simplefwi.misfit import (
    DataMisfit,
    RegMisfit,
    MisfitFunction,
)

from simplefwi.optimization import (
    BBiter,
    CGiter,
    deterministicAlpha,
)

# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core forward modeling
    "build_helmholtz_matrix",
    "build_interpolation_operator",
    "build_sensitivity_kernel",
    "ForwardModeler",
    "ForwardSolver",  # Backward compatibility
    "JacobianOperator",
    # Misfit functions
    "DataMisfit",
    "RegMisfit",
    "MisfitFunction",
    # Optimization
    "BBiter",
    "CGiter",
    "deterministicAlpha",
]
