# SimpleFWI

<p align="center">
  <img src="asset/SimpleFWI_logo.png" alt="SimpleFWI Logo" width="400"/>
</p>

**Performance-grade frequency-domain Full Waveform Inversion (FWI)**

SimpleFWI is a high-performance Python package for frequency-domain Full Waveform Inversion of seismic data. Built on PyLops for matrix-free operators, it supports multiple backends (NumPy, CuPy, JAX) for CPU and GPU acceleration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Primary Objective

**Ultra-fast computation with C-level performance** (50-100× speedup) + **optimal memory efficiency** for large-scale problems (N > 1e7 parameters).

## ✨ Features

- **Frequency-domain FWI**: Solves Helmholtz equation for acoustic wave propagation
- **Matrix-free operators**: Memory-efficient via PyLops `LinearOperator`
- **Backend-agnostic**: Automatic support for NumPy (CPU), CuPy (CUDA), JAX (Metal/CUDA)
- **Adjoint-state gradients**: Efficient gradient computation using imaging condition
- **Iterative optimization**: Barzilai-Borwein, Conjugate Gradient
- **Apple Silicon support**: Native Metal GPU acceleration via JAX
- **SOLID architecture**: Clean abstractions for extensibility

## 🚀 Quick Start

### Installation

**For Apple Silicon (M1/M2/M3)**:
```bash
git clone https://github.com/izzatum/SimpleFWI.git
cd SimpleFWI
conda create -n simplefwi python=3.10
conda activate simplefwi
pip install -e ".[complete-metal]"
```

**For NVIDIA GPU (CUDA)**:
```bash
pip install -e ".[complete-cuda]"
```

**CPU only**:
```bash
pip install -e ".[complete]"
```

### Basic Usage

```python
import numpy as np
from simplefwi import ForwardSolver, DataMisfit, BBiter

# Define model
model = {
    'h': [10.0, 10.0],          # Grid spacing [dz, dx] in meters
    'n': [100, 100],            # Grid dimensions [nz, nx]
    'f': [2.0, 3.0, 4.0],       # Frequencies in Hz
    'xs': [500],                # Source x-coordinates
    'zs': [10],                 # Source z-coordinates
    'xr': np.arange(0, 1000, 20),  # Receiver x-coordinates
    'zr': 10*np.ones(50),       # Receiver z-coordinates
    'q': np.eye(1),             # Source wavelet matrix
}

# Create slowness-squared model (m = 1/v²)
m_true = (1/2.0)**2 * np.ones((10000, 1))  # True model
m0 = (1/2.5)**2 * np.ones((10000, 1))      # Initial guess

# Generate synthetic data
Fm = ForwardSolver(model)
Dobs, J = Fm.solve(m_true)

# Run inversion
misfit = DataMisfit(Dobs, model)
history, m_inv, g = BBiter(misfit, m0, tol=1e-3, maxit=10)

# Visualize result
v_inv = 1.0 / np.sqrt(m_inv.reshape(100, 100))
```

## 📦 Package Structure

```
SimpleFWI/
├── simplefwi/              # Main package
│   ├── __init__.py         # Package exports
│   ├── core.py             # Forward modeling & Jacobian
│   ├── misfit.py           # Objective functions
│   └── optimization.py     # Optimizers (BB, CG)
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── validation/         # Mathematical correctness tests
├── data/                   # Example datasets
├── Example_*.ipynb         # Tutorial notebooks
└── pyproject.toml          # Package configuration
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=simplefwi --cov-report=html

# Run only fast tests (skip slow benchmarks)
pytest tests/ -m "not slow"

# Run GPU tests (if available)
pytest tests/ -m gpu
```

## 📚 Examples

1. **Example_modelling.ipynb**: Forward modeling basics
2. **Example_1.ipynb**: Circular reflector (Green's function reciprocity)
3. **Example_2.ipynb**: Basic inversion with BB and CG
4. **Example_3.ipynb**: Marmousi model (realistic benchmark)

## 🛠️ Development Status

**Current**: v0.1.0 - Research-quality implementation

**Roadmap** (see `.github/copilot-instructions.md`):
- **Phase 1**: GPU acceleration (10-50× speedup) ⏳
- **Phase 2**: JIT compilation (2-5× speedup) ⏳
- **Phase 3**: Iterative solvers for N > 1e6 ⏳
- **Phase 4**: Parallel processing (2-4× speedup) ⏳
- **Phase 5**: Matrix-free implementations ⏳
- **Phase 6**: SOLID refactoring ⏳
- **Phase 7**: Hessian operators (Gauss-Newton, Full) ⏳
- **Phase 8**: Advanced optimizers (ADAM, L-BFGS, Newton-CG) ⏳
- **Phase 9**: Enhanced absorbing boundaries (sponge, Rayleigh) ⏳
- **Phase 10**: Complete documentation & CI/CD ⏳

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- SOLID principles & performance guidelines
- Testing requirements
- Code style & documentation standards

## 📖 Documentation

- **API Reference**: NumPy-style docstrings in source code
- **Examples**: Jupyter notebooks in repository root

## 🔧 Dependencies

**Core**:
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- PyLops ≥ 2.3.0
- Matplotlib ≥ 3.7.0

**GPU Acceleration** (optional):
- CuPy ≥ 12.0.0 (NVIDIA CUDA)
- JAX[metal] ≥ 0.4.20 (Apple Silicon)

**Performance** (optional):
- Numba ≥ 0.58.0
- Cython ≥ 3.0.0

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [PyLops](https://pylops.readthedocs.io/) for matrix-free operators
- Backend abstraction pattern follows PyLops conventions
- Inspired by frequency-domain FWI research community

## 📧 Contact

- **Author**: Muhammad Izzatullah
- **GitHub**: [@izzatum](https://github.com/izzatum)
- **Issues**: [GitHub Issues](https://github.com/izzatum/SimpleFWI/issues)

---

**Note**: SimpleFWI is under active development. The API may change as we implement the performance-focused roadmap.
