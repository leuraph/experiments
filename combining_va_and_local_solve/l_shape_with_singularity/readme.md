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