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
on each space, performs full sweeps of local solves until
$$
E(u^n) - E(u^{n+1}) < \alpha \frac{ E(u^{n_{\text{min}}}) - E(u^{n+1}) }{n - n_{\text{min}}},
$$
i.e. until the global energy gain flattens off.

## Experiment 02
On each space, performs full sweeps on all active elements until
the fraction of inactive elements reaches a certain threshold, then we refine
the space using element-based VA and start all over again.

On each element $T$, we monitor its contribution
$E_T$ to the overall energy $E = \sum_T E_T$, i.e.
$$
E_T(u^u_N) := \int_T
\Bigg(
    \frac{1}{2} \big|\nabla u^n_N(x)\big|^2
    - f(x)u^n_N(x)
\Bigg) ~\mathrm{d} x.
$$
For the deactivation of the elements, we would like to use a localized
version of the stopping criterion above, i.e. deactivate the element if
its energy decay flattens off.
However, there is a small caveat.
As we locally solve on each element, it might happen that,
for neighbouring elements, the energy increases.
Therefore, we can not guarantee monotonicity for the energy contributions $E_T$.
Instead, we keep track of the summed absolute values, i.e.

$$
\widetilde{\Phi}^n_T
:=
\sum_{k=n_{\text{min}}}^n
\left|
E_T(u^{n-1}) - E_T(u^{n})
\right|,
\quad
n > n_{\text{min}},
$$

and mark all elements $T$ as deactivated that meet
$$
|E_T(u^{n-1}) - E_T(u^{n})|
<
\alpha \frac{ \widetilde{\Phi}^n_T}{n - n_{\text{min}}},
\quad
n>n_{\text{min}}
$$

Note that any element $T$ might get marked active after it has already been
marked deactivated as changes in any neighbouring element also changes $E_T$.

## Experiment 03
This is a copy of experiment02 with the only difference that elements
that once have been marked deactivated do not get activated again.
Hence, the number of active elements is monotonically decreasing.