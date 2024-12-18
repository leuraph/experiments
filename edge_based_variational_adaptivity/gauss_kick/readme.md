# Setup

Here, we consider numerical experiments, where we impose the exact solution to be given by
$$
u(x,y) = x(x-1)y(y-1)\exp( - \sigma_x (x-\mu_x)^2 - \sigma_y (y-\mu_y)^2 ).
$$

The energy norm squared $\|u\|_a^2$ is approximated using `experiment_notes.ipynb`.
Then, using Galerkin orthogonality, we find
$$
\|u - u_h^N\|_a^2 = \|u\|_a^2 - \|u_h^N\|_a^2,
$$
where $\|u_h^N\|_a^2 = a(u_h^N, u_h^N) = (u_h^N)^\top \mathbf{A} u_h^N$.

## Experiment 01
Exact solving and refining using Edge based variatioinal adaptivity.
To calculate the energy drop per edge, we refine a single non-boundary edge
and calculate the solution to the corresponding 2x2 system of linear equations.
Note that this experiment is very slow as we explicitly refine a single edge
and rebuild the whole stiffness matrix and the right hand side vector for each edge.

## Experiment 02
Same experiment as experiment-01.
However, this experiment runs faster as we extract the local patch of
two elements before refining the shared edge.

## Experiment 03
Global CG iterations stopped by comparison of global energy losses of EVA and n CG steps.
Note that this experiment makes use of `custom_callback.py` as we need not to interrupt the
global CG iterations when checking for convergence.
In this experiment, we
- perform `n` global cg update steps
- perform edge based variational adaptivity to get new values on non-boundary edges
- in parallel, perform `n` more global CG step
- compare the corresponding global energy gains and, depending on this comparison, decide whether to refine the mesh or not

## Experiment 4
global CG iterations stopped by comparison of current energy losses and overall energy loss on current space. Namely, we stop the CG iterations and refine, as soon as we have
$$
E(u^{n-1}) - E(u^n) \leq \alpha(n) \Big(E(u^0) - E(u^n)\Big),
$$
where $\alpha(n) =\text{const}. \in (0, 1)$ or (even stronger) $\alpha(n) \propto n^{-1} \in (0, 1)$.
This ciretrion is motivated by [HAW23].

# Error Calculations

## Experiment 1/2
Here we calculate the exact Galerkin Solution on each mesh.
Hence, to calculate the energy norm error $\|u - u^N_h\|_a$ we can make use
of Galerkin Othogonality, i.e.
$$
\|u - u^N_h\|^2_a = \|u\|_a^2 - \|u^N_h\|_a^2.
$$
So, to compute the energy norm error, it is enough to calculate
the energy norm of the solution $u$ once
and the energy norm squared for each Galerkin solution.

## Experiment 3
Here we iteratively approximate the Galerkin solution on each mesh.
Hence, we can not make use of Galerkin Orthogonality and therefore
must compute (approximate) the integral
$$
\|u - u_n^N\|_a^2 = \int_{\Omega} |\nabla(u - u_n^N)|^2 ~ \mathrm{d}x.
$$

## Scripts

- `edge_based_variational_adaptivity/gauss_kick/calc_errors_for_exp.py`:
  On each subspace, i.e. for every number of degrees of freedom,
  calculates the energy norm error squared for the exact galerkin solution
  and the last iterate.
- `edge_based_variational_adaptivity/gauss_kick/calculate_energy_norms_squared.py`:
  On each subspace, i.e. for every number of degrees of freedom,
  calculates the energy norm error squared for `solution.pkl`
- `edge_based_variational_adaptivity/gauss_kick/plot_energy_errors_galerkin_decay.py`:
  Plot the energy norm error decay for the exact Galerkin solutions.
- `edge_based_variational_adaptivity/gauss_kick/plot_energy_norm_errors_for_exp.py`:
  Plot the energy norm error decay for both the Galerkin solution as well as the last iterate.
