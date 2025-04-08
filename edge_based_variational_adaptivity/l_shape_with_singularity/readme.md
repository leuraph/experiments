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

## Experiment 10

# Combining Arioli's stopping criterion [1] with a new potential upper bound or approximation
Based on sole numerical experiments (for now), it is observed
that the following iteration can produce upper bounds on the Galerkin
solution's energy norm.
Namely, let $u^n$ be the current approximation of $u_h$ obtained after $n$ CG steps.
then, we perform the following updates
$$
\begin{aligned}
    w_0 &= u_n, \\
    d_0 &= r_0, \\
    w_{j+1} &= w_j + \alpha(w_j, d_j) d_j, \\
    d_{j+1} &= H(r_{j+1}) d_{j},
\end{aligned}
$$
where $H(v)$ is the Householder reflector associated to $v$,
$\alpha(w_j, d_d) := \frac{r_j^\top d_j}{\|d_j\|_2^2}$
ensuring that we always stay on the level set $\mathcal{M}_{w_0}$.
The hypothesis is that, after "enough" iterations, we visit
points for which we have $\|u_h\|_a \leq \|w_n\|_a$, i.e.
an upper bound for the Galerkin solution's energy norm.

The following experiments are based on the update procedure
as described above.
Reqriting Arioli's stopping criterion yields
$$
(1-\gamma^2) \|u_h\|^2_a \leq -2 E(u_n)
$$

## Experiment 11
Given some delay value $d \in \mathbb{N}$,
we postulate
$$
\|u_h\|_a^2 \leq \max_{n \in \{0, \dots, d\}} \|w_n\|^2_a
$$

## Experiment 12
Given some delay value $d \in \mathbb{N}$,
we define
$$
\overline{w} := 1/d \sum_{n=1}^d w_n
$$
and postulate
$$
\|u_h\|_a^2 \approx \| \overline{w} \|^2_a
$$

## Experiment 13
Same as experiment 11 but the delay value $d$ is
dynamically set to the corresponding number of
degrees of freedom, i.e.
$d = n_{\text{DOF}}$.

## Experiment 14
Same as experiment 12 but the delay value $d$ is
dynamically set to the corresponding number of
degrees of freedom, i.e.
$d = n_{\text{DOF}}$.

## Experiment 15 (monitor relative changes in energy and energy norm)
This experiment is done in order to produce some approximations
$u_N^n$ used to plot and monitor the relative changes in
(i) the energy norm squared and
(ii) the energy itself, i.e.
$$
\begin{aligned}
(i)~~M_1^n &:= \left| \frac{\|u_N^{n-1} - u_N^n\|_a^2}{\|u_N^n\|_a^2} \right|, \\
(ii)~~M_2^n &:= \left| \frac{E(u_N^{n-1}) - E(u_N^n)}{E(u_N^n)} \right|.
\end{aligned}
$$

Note that the experiment itself drops the approximations $u_N^n$, their energies and energy norms squared.

This is done in the following way.
Start with a relatively fine mesh $\mathbb{V}_0$, solve exactly to get $u^h_0$, refine with EVA to get an initial mesh $\mathbb{V}_1$ and canonically embed (linear interpolation) $u^h_{0} \mapsto u^0_1 \in \mathbb{V}_1$.
Then, on each mesh, perform a pre-defined number of CG iterations `n_cg_iterations`, use the last approximation to perform EVA.
Repeat on each mesh until the number of DOFs reaches a pre-defined maximum.

## Experiment 16
This experiment is a copy of experiment 08
with the only adaption that we use a diagonal preconditioner
$$
M := \text{diag} A.
$$
In our handwritten notes, we have shown that,
despite using the preconditioner $M$,
the energy version Arioli's stopping criterion [1]
does not need to be changed.

# References
- [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.