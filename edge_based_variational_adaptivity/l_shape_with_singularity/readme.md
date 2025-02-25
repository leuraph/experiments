# Experiments

### Setup
This subdirectory considers the problem $- \Delta u = 1$
with homogeneous boundary conditions on the L-shape domain as given by
$\Omega := [-1, 1]\times[0, 1] \cup [-1, 0]\times[-1, 0]$.

### Energy norm error
Note that the analytical solution of this problem remains unknown.
However, we do have an estimate of the value $\| u \|_a^2$ at hand,
i.e. we received, via mail (Patrick Bammer), the value
$\| u \|_a^2 \approx 0.214075802220546$.
Using Galerkin Orthogonality for the Galerkin solutions yields
$\|u - u_h\|_a^2 = \|u\|_a^2 - \|u_h\|_a^2$

## Experiment 01
On each mesh, calculate the exact Galerkin solution $u^h_N$
and refine the mesh adaptively using EVA with a prescribed value of
$\theta$ and the exact Galerkin solution $u_N^h$ found.

## Experiment 05
This experiment performs global batched CG iterations until the condition
$E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}$
is met.
Then, EVA is performed using the last iterate.

## Experiment 06
This experiment performs global batched CG iterations until the condition
$E(u^{n-1}) - E(u^{n}) < \frac{\alpha}{n_{\text{DOF}}}$
is met.
Then, EVA is performed using the last iterate.

## Experiment 07
This experiment performs global batched CG iterations until the condition
$\|u_N^n - u_N^h\|_a^2 \leq \alpha \|u_N^h - u\|_a^2$
is met.
Then, EVA is performed using the last iterate.
This serves as a sort of sanity check, i.e. we may see if EVA still
reaches optimal convergence (for the Galerkin solutions) if we set $\alpha \approx 1$.

## Experiment 08
This experiment performs global CG iterations until an energy-based version
of Ariolis [1] stopping criterion is met, i.e.
$$
E(u^n) - E(u^{n+d}) \leq \left(\frac{- \alpha}{n_{\text{Dof}}}\right) E(u^n),
$$
where $\alpha$ denotes an additional fudge parameter, not mentioned in [1].
Then, EVA is performed using the last iterate $u^{n+d}$,
where $d$ denotes the delay.

## Experiment 09
This experiment is a sanity check for Ariolis [1] stopping criterion.
Namely, this experiment performs global CG iterations until an exact implementation
of Ariolis [1] stopping criterion without any approximations is met, i.e.
$$
\|u_h - u_h^{n^\star}\|_a^2 \leq \frac{\alpha}{n_{\text{Dof}}} \|u_h\|_a^2,
$$
where $\alpha$ denotes an additional fudge parameter, not mentioned in [1].
Then, EVA is performed using the last iterate $u^{n+d}$,
where $d$ denotes the delay.

# References
- [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.