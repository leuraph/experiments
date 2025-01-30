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