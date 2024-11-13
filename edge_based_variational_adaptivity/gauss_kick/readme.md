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