"""Implicit Helmholtz differentiation, including derivatives of the adjoint.

Numerical solves are opaque primitives. Their derivatives use another implicit
solve, so neither SciPy nor Krylov iteration histories enter the autograd graph.
Real model gradients use PyTorch's real-loss complex-cotangent convention.
"""
from __future__ import annotations

import torch


class _ImplicitSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, rhs, omega, operator, solver, adjoint):
        solution, report = solver.solve(operator, m, omega, rhs, adjoint=adjoint)
        if not report.converged:
            from .solvers.base import SolverFailure
            raise SolverFailure('Implicit solve requires a converged solution', report)
        ctx.save_for_backward(m, rhs, solution)
        ctx.omega, ctx.operator, ctx.solver, ctx.adjoint = omega, operator, solver, adjoint
        return solution

    @staticmethod
    def backward(ctx, grad_solution):
        m, rhs, saved_solution = ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        operator, omega, solver = ctx.operator, ctx.omega, ctx.solver
        # Recompute the primal with its implicit graph when differentiating the
        # backward. Saving only the forward's detached solution loses du/dm.
        solution = (_ImplicitSolve.apply(m, rhs, omega, operator, solver, ctx.adjoint)
                    if higher_order else saved_solution)
        dual = _ImplicitSolve.apply(m, grad_solution, omega, operator, solver,
                                    not ctx.adjoint)
        grad_m = None
        if ctx.needs_input_grad[0]:
            with torch.enable_grad():
                # A fresh perturbation isolates the partial model derivative
                # while retaining solution/dual dependencies in its VJP graph.
                dm = torch.zeros_like(m, requires_grad=True)
                if ctx.adjoint:
                    product = operator.model_derivative(dm, omega, solution.conj()).conj()
                else:
                    product = operator.model_derivative(dm, omega, solution)
                grad_m, = torch.autograd.grad(product, dm, grad_outputs=-dual,
                                              create_graph=higher_order)
        grad_rhs = dual if ctx.needs_input_grad[1] else None
        return grad_m, grad_rhs, None, None, None, None


def implicit_solve(m: torch.Tensor, rhs: torch.Tensor, omega: float, operator,
                   solver, *, adjoint: bool = False) -> torch.Tensor:
    """Solve A(m,omega)u=rhs with exact implicit first/second derivatives.

    ``m`` is physical real slowness squared [z,x]. ``rhs`` and the result are
    complex compact vectors [...,n_dof]. ``omega`` is positive rad/s and fixed
    during differentiation. ``adjoint=True`` solves A^H instead. The operator
    has model-linear diagonal mass dependence, including PML extension.
    """
    return _ImplicitSolve.apply(m, rhs, float(omega), operator, solver, bool(adjoint))
