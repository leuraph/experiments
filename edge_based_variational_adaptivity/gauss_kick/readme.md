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
global CG iterations stopped by global energy drop.
In this experiment, we
- perform `n` global cg update steps
- perform edge based variational adaptivity to get new values on non-boundary edges
- in parallel, perform one more global CG step
- compare the corresponding global energy gains and, depending on this comparison, decide whether to refine the mesh or not

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