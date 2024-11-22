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
two elements before refining the mesh.

## Experiment 03