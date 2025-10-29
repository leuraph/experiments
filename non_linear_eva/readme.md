# Summary

This folder contains experiments for the
nonlinear edge-based variational adaptivity algorithm (EVA)
in companion with the nonlinear CG algorithm using
default and custom stopping criteria for
variational problems of the form

$$
\int_\Omega
\left\langle
    \mathbf A (\nabla u (\mathbf x)),
    \nabla v (\mathbf x)
\right\rangle~\mathrm d \mathbf x
+
\int_\Omega
\phi\big(u(\mathbf x)\big) v(\mathbf x)
~\mathrm d \mathbf x
=
\int_\Omega
f(\mathbf x) v(\mathbf x)
~\mathrm d \mathbf x,
$$

where $\phi \in C^1(\mathbb R)$ is the
strictly monotonically increasing
nonlinearity of the problem.

## Problem 1
- Domain: L-shape
- $\mathbf A = \mathrm{id}_{2\times 2}$
- $\phi(u) := u^3$
- $\Phi(u) = u^4 / 4$
- $\phi'(u) = 3 u^2$
- $f \equiv 1$

## Problem 2
- Domain: L-shape
- $\mathbf A = \mathrm{id}_{2\times 2}$
- $\phi(u) := u |u|$
- $\Phi(u) = u |u|^2 / 3$
- $\phi'(u) = 2 |u|$
- $f \equiv 1$

## Problem 3
- Domain: L-shape
- $\mathbf A = \mathrm{id}_{2\times 2}$
- $\phi(u) := \exp (u) - 1$
- $\Phi(u) = \exp(u) - u$
- $\phi'(u) = \exp (u)$
- $f \equiv 1$

## Problem 4 (imposed solution)
The following problem is taken from
[https://arxiv.org/abs/2504.11292, chapter 4.2].

- Domain: L-shape
- $\mathbf A = \mathrm{id}_{2\times 2}$
- $\phi(u) := u^3$
- $\Phi(u) = \frac{u^4}{4}$
- $\phi'(u) = 3 u^2$
- $u(x, y) := r^{-4/3} xy (1-x^2)(1-y^2), \quad r:= \sqrt{x^2 + y^2}$
- $f(x, y)$ is chosen accordingly