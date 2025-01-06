# Summary

In this experiment, we investigate hot the slope of
$\|u_h - u\|^2_a$ vs. $n_{\text{DOF}}$,
where $u_h$ denotes the Galerkin solution and $u$ the exact solution,
behaves, if we randomly select a certain percentage of elements for refinement.

## Setup
The exact solution is imposed to be a Gauss Bump of the form
$$
u(x,y) = x(x-1)y(y-1)\exp( - \sigma_x (x-\mu_x)^2 - \sigma_y (y-\mu_y)^2 ).
$$

The energy norm squared $\|u\|_a^2$ is approximated using `experiment_notes.ipynb`.
Then, using Galerkin orthogonality, we find
$$
\|u - u_h^N\|_a^2 = \|u\|_a^2 - \|u_h^N\|_a^2,
$$
where $\|u_h^N\|_a^2 = a(u_h^N, u_h^N) = (u_h^N)^\top \mathbf{A} u_h^N$.