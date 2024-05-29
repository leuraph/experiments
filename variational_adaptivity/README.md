# Experiment Notes

## L-Shape with singularity
Looking at the results, we find that the `variational adaptivity`
refines a boundary-layer when there should be none.
I think that this might be, as in the VA code version used,
locally solving the problem does not yet respect any boundary conditions,
i.e. when we solve for the potential energy gain on an element
touching the boundary, we do not impose any boundary conditions to
the local solution.
This however, is just a gut-feeling and needs to be checked in further
experiments using new versions of the code.

## Square with singularity
Solving $- \Delta u = 1$ on $(0, 1)^2$ with homogeneous Dirichlet 
boundary conditions. For the exact solution, we do not have an
exact analytical representation, but we do have the energy norm
in terms of an Eigenfunction expansion, see [1].

## L-shape with singularity revisited
A mere copy of `L-Shape with singularity`, performed with an
updated version of the VA algorithm.
Namely, in this experiment, when solving the local 4x4 matrix equation
to get the potential energy gain, we respect any Dirichlet boundary conditions.

# References

- [1] 10.48550/arXiv.2311.13255