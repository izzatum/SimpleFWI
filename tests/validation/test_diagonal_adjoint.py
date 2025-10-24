"""
Analyze the adjoint formula more carefully.

For the Jacobian:
  Forward: J @ dm = Pr @ A^{-1} @ (-G @ dm)
  Adjoint: J^H @ dd = ???

Let's derive it step by step:
  <dd, J @ dm> = <dd, Pr @ A^{-1} @ (-G @ dm)>
               = <Pr^H @ dd, A^{-1} @ (-G @ dm)>
               = <A^{-H} @ Pr^H @ dd, -G @ dm>
               = <-G^H @ (A^{-H} @ Pr^H @ dd), dm>

So: J^H @ dd = -G^H @ (A^{-H} @ Pr^H @ dd)

The question is: what is G^H for a diagonal matrix G?

For diagonal G with entries g_i:
  G @ x has components: (G @ x)_i = g_i * x_i

The adjoint satisfies:
  <y, G @ x> = <G^H @ y, x>

For complex inner product <a, b> = sum(conj(a_i) * b_i):
  <y, G @ x> = sum(conj(y_i) * g_i * x_i)
  <G^H @ y, x> = sum(conj((G^H @ y)_i) * x_i)

For these to be equal:
  conj((G^H @ y)_i) = conj(y_i) * g_i
  (G^H @ y)_i = conj(conj(y_i) * g_i) = y_i * conj(g_i)

So: G^H @ y = diag(conj(g)) @ y

This means for diagonal matrix: G^H = conj(G) (just conjugate the diagonal)
"""

import numpy as np
from scipy import sparse

# Test this theory
g = np.array([1 + 2j, 3 + 4j, 5 + 6j])
G = sparse.diags(g)

x = np.array([1, 2, 3])
y = np.array([4 + 1j, 5 + 2j, 6 + 3j])

# Forward
Gx = G @ x
print("G @ x =", Gx)

# What should G^H @ y be?
GH_theory = sparse.diags(np.conj(g))
GHy_theory = GH_theory @ y
print("Theory: G^H @ y =", GHy_theory)

# Verify adjoint test
lhs = np.vdot(y, Gx)
rhs = np.vdot(GHy_theory, x)
print(f"\n<y, G@x> = {lhs}")
print(f"<G^H@y, x> = {rhs}")
print(f"Match? {np.allclose(lhs, rhs)}")

# Now test with .conj().T
GH_scipy = G.conj().T
GHy_scipy = GH_scipy @ y
print(f"\nUsing .conj().T: G^H @ y = {GHy_scipy}")
print(f"Match theory? {np.allclose(GHy_scipy, GHy_theory)}")

# Verify
rhs2 = np.vdot(GHy_scipy, x)
print(f"<G^H@y, x> using .conj().T = {rhs2}")
print(f"Match? {np.allclose(lhs, rhs2)}")
