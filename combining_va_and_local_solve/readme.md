# Summary

## Experiment 1
In this experiment, we do the following.

- initialize a random solution which is equal to zero on the boundary of the domain.
- loop over all alements $T \in \mathcal{T}$.
  - compute the local increment $\Delta x$ and its corresponding energy gain $\Delta E^{\text{local}}$
  - compute the possible energy gain with variational adaptivity, i.e locally red-refine and solving
- put the energy differences in a vector $\Delta E$, where we save
  $\Delta E_j = \max \{ \Delta E^{\alpha \text{local}}_j, \Delta E^{\text{VA}}_j \}$,
  where $\alpha$ is a fudge parameter.
- With $\Delta E$, perform a dörfler marking
- perform all marked local updates such that updates corresponding to biggest local energy drop come last
- perform all marked refinements using collective `NV` refinement

## Eperiment 2
In this experiment, we do the following.

- initialize a random solution which is equal to zero on the boundary of the domain.
- loop over all alements $T \in \mathcal{T}$.
  - compute the local increment $\Delta x$ and its corresponding energy gain $\Delta E^{\text{local}}$
  - compute the possible energy gain with variational adaptivity, i.e locally red-refine and solving
- put the energy differences in a vector $\Delta E$, where we save
  $\Delta E_j = \max \{ \Delta E^{\alpha \text{local}}_j, \Delta E^{\text{VA}}_j \}$,
  where $\alpha$ is a fudge parameter.
- perform all local updates corresponding to entries in $\Delta E$,
  i.e. for all $T \in \mathcal{T}$,where $\alpha \Delta E^{\text{local}}_j > \Delta E^{\text{VA}}_j$
  such that updates corresponding to biggest local energy drop come last.
- Perform a dörfler marking for all elements marked for refinement
- perform all marked refinements using collective `NV` refinement