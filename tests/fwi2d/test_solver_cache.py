"""Numeric factors have value-verified identity and a bounded explicit lifetime."""
from dataclasses import replace
import torch
from simplefwi.fwi2d import SolverConfig
from simplefwi.fwi2d.solvers import DirectSolver
from test_direct_solver import TinyOperator, model, rhs


def test_rhs_batches_and_adjoint_reuse_one_factorization():
    solver, operator, m = DirectSolver(SolverConfig()), TinyOperator(), model()
    solver.solve(operator, m, 7., rhs(1))
    factor_bytes = solver.cache_bytes
    solver.solve(operator, m.clone(), 7., rhs(12))
    solver.solve(operator, m, 7., rhs(2), adjoint=True)
    assert solver.factorization_count == 1
    assert solver.cache_entries == 1 and solver.cache_bytes == factor_bytes


def test_model_mutation_frequency_and_pml_changes_invalidate():
    solver, operator, m = DirectSolver(SolverConfig()), TinyOperator(), model()
    solver.solve(operator, m, 7., rhs())
    with torch.no_grad():
        m[0, 0] += .1
    result, _ = solver.solve(operator, m, 7., rhs())
    torch.testing.assert_close(operator.apply(m, 7., result), rhs())
    solver.solve(operator, m, 8., rhs())
    operator.boundary = replace(operator.boundary, sigma_max=200.)
    solver.solve(operator, m, 8., rhs())
    assert solver.factorization_count == 4


def test_cache_lru_eviction_is_bounded():
    operator, m = TinyOperator(), model()
    probe = DirectSolver(SolverConfig())
    probe.solve(operator, m, 7., rhs())
    budget = probe.cache_bytes + 16
    solver = DirectSolver(SolverConfig(memory_budget_bytes=budget))
    solver.solve(operator, m, 7., rhs())
    solver.solve(operator, m, 8., rhs())
    assert solver.cache_bytes <= budget
    assert solver.cache_entries == 1
    solver.solve(operator, m, 7., rhs())
    assert solver.factorization_count == 3


def test_oversized_factor_is_used_but_not_retained():
    solver, operator = DirectSolver(SolverConfig(memory_budget_bytes=1)), TinyOperator()
    solver.solve(operator, model(), 7., rhs())
    assert solver.cache_bytes == 0 and solver.cache_entries == 0


def test_explicit_evaluation_scope_clears_numeric_state_even_on_error():
    solver, operator = DirectSolver(SolverConfig()), TinyOperator()
    try:
        with solver.evaluation():
            solver.solve(operator, model(), 7., rhs())
            assert solver.cache_entries == 1
            raise RuntimeError('trial evaluation abandoned')
    except RuntimeError:
        pass
    assert solver.cache_entries == 0 and solver.cache_bytes == 0
    solver.solve(operator, model(), 7., rhs())
    solver.clear_cache()
    assert solver.cache_entries == 0


def test_grid_origin_changes_configuration_identity():
    solver, operator, m = DirectSolver(SolverConfig()), TinyOperator(), model()
    solver.solve(operator, m, 7., rhs())
    operator.grid = replace(operator.grid, origin_m=(0., 0.))
    solver.solve(operator, m, 7., rhs())
    assert solver.factorization_count == 2
