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
- $u(x, y) := 2 r^{-4/3} xy (1-x^2)(1-y^2), \quad r:= \sqrt{x^2 + y^2}$
- $f(x, y)$ is chosen accordingly

# Post-Processing
As we are considering nonlinear problems in this
folder, the post-processing turns out to be way more difficult,
if we want to plot not only the energy error
$\mathrm{E}(u^\star_N) - \mathrm{E}(u)$,
but also the energy norm error
$\|u^\star_N - u\|^2_a$.

As for problems 1,2, and 3,
the exact solutions are unknown,
we rely on a reference solution
$\mathbb P_1 (\mathcal T_h) \ni \tilde u \approx u$,
that approximates the exact solution
as well as possible.
Reference solutions are computed as follows
and described in [this paper](https://arxiv.org/abs/2504.11292).

1. Generate a graded mesh with the script
   `graded_meshes/refinement_script.m`
2. Compute the reference solution on the graded mesh using the script `graded_solver.py`.

## Problem 1
For this problem, we want to plot the energy norm error,
despite not having access to the exact solution $u$.
Following the above procedure, we do have a reference solution at hand.
However, as the reference solution was computed on
a pre-computed graded mesh, it does not only
not live in the same FEM-space as $u_N^\star$,
but the two meshes do, in general, not have anything in common.
The problem becomes apparent when remembering
one needs to evaluate the term
$a(\tilde u, u_N^\star)$ in order to compute
the energy norm error.
As we may assume the graded mesh has more degrees of
freedom than the finest mesh produced by EVA,
we assume the finest graded mesh available
resolves the singularity much better.
Motivated by this observation, we provide an
implementation of the mapping that evaluates
any FEM function on a given set of coordinates
via the `python` package `p1afempy`,
i.e. the function `p1afempy.solvers.evaluate_on_coordinates`
provides the mapping
$$
u_N^\star \mapsto I_h(u^\star_N) := \sum_{z \in \mathcal V_h}
u_N^\star(z) \phi_{z}.
$$
Once the "transfer" is finished
(for many degrees of freedom this may take a while),
the computation of the energy norm error
$\|\tilde u - I_h(u^\star_N)\|_a^2$
becomes very easy as both FEM functions
live on the same graded mesh now.

### Tool-Chain

1. Compute the reference mesh and a reference solution on it.
2. Do the experiment(s) `experiments.py`.
3. For the results, perform the mesh transfer.
4. Compute the energy norm errors:
   `compute_error_norms_squared.py`
   (This script performs the tedious mesh transfer
   but does not save the resulting transferred solutions).
5. Plot the energy norm errors.

## Problem 2 / 3

For these problems, we merely plot the energy errors
$\mathrm E (u^\star_N) - \mathrm E (\tilde u)$,
where $\tilde u$ is the reference solution computed
on a graded mesh as described above.

### Tool-Chain

1. Compute the reference mesh and a reference solution on it.
2. Do the experiment(s) `experiments.py`.
3. Compute the energies from experiments: `compute_energies.py`.
4. Plot the energy errors: `plot_energy_errors`
   (This script computes the reference energy on the way,
   using the reference solution on the reference mesh).

## Problem 4
This problem is constructed such
that exact solution $u : \Omega \to \mathbb{R}$ is known.
Nevertheless, the numerical evaluation of
the energy norm error turns out to be non-trivial.
Consider
$$
\|u_N^\star - u\|_a^2
=
a(u_N^\star, u_N^\star) + a(u, u) - 2a(u, u_N^\star).
$$
The **first term** is easily computed with the stiffness matrix.
To evaluate the **second term**, i.e.
$$
a(u, u) = 
\int_\Omega 
\underbrace{
    \langle
        \nabla u(\mathbf{x}), \nabla u(\mathbf{x})
    \rangle
}_{=: I(\mathbf{x})}
~\mathrm{d}\mathbf{x},
$$
we compute the integrand $I(\mathbf{x})$ symbolically
and approximate the integral $\int_\Omega I(\mathbf{x}) ~\mathrm{d} \mathbf{x}$ numerically by integrating the symbolic expression $I(\mathbf{x})$
over the finest _graded mesh_ available.
> The graded meshes may be computed using the accompanying script `graded_meshes/refinement_script.m`. Unfortunately, this script is available only as MATLAB implementation. Feel free to write down your own implementation, if needed.

The **third term** is re-written, using the fact that
$u$ solves the weak equation
$$
\int \nabla u \nabla v
+ \int_\Omega \phi(u) v
= \int_\Omega f v
\quad
\forall v \in\mathrm{H}^1_0 (\Omega),
$$
i.e. we have
$$
- 2 a(u, u_N^\star)
=
-2
\left[
    \int_\Omega f u^\star_N - \int_\Omega \phi(u) u^\star_N
\right],
$$
which, given that we know both
$\phi : \mathbb R \to \mathbb R$
and
$u: \Omega \to \mathbb R$ analytically,
can be both integrated using the same routine.

An implementation of all of these computations
is found in the post-processing script
`compute_energy_norm_errors_squared_problem-4.py`.

### Tool-Chain

1. Do the experiment(s) `experiments.py`
2. Compute a reference value for $\|u\|_a^2$
   (using pre-computed graded meshes and numerical integration)
   with `energy_norm_problem_4.ipynb`.
   If needed, change the hard-coded reference value in the script
   `compute_energy_norm_errors_squared_problem-4.py`.
3. Compute the energies norm errors:
    `compute_energy_norm_errors_squared_problem-4.py`.
    Given a reference value for $\|u\|_a^2$,
    this script performs the post-processing as described above.
3. Plot the energy norm errors: `plot_energy_norm_errors.py`