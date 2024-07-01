# Summary

In these experiments, we consider the following algorithms.

## ALGORITHM-1
- sweep over all elements $T \in \mathcal{T}$
  - using a local solver, compute the local increment $d_n^{(T)}$
  - update the local element's degrees of freedom (DOF) using $d_n^{(T)}$
- repeat for $N$ sweeps 

## ALGORITHM-2
- sweep over all elements $T \in \mathcal{T}$
  - using a local solver, compute the local increment $d_n^{(T)}$
  - compute the local energy difference $\Delta E^{(T)}$
- use dörfler marking on elements with respect to all local energy differences
- sort the marked elements and local increments in ascending order
  such that elements with highest energy difference come last
- loop over all local increments $d_n^{(T)}$ corresponding to marked elements,
  collecting them in one global increment $d_n$, each time overwriting
  any possible "old" increment to make sure that for any DOF we use the increment corresponding
  to the biggest local energy difference
- perform one global update $x_{n+1} = x_n + d_n$
- repeat for $N$ sweeps

## ALGORITHM-3
- sweep over all elements $T \in \mathcal{T}$
  - using a local solver, compute the local increment $d_n^{(T)}$
  - compute the local energy difference $\Delta E^{(T)}$
- use dörfler marking on elements with respect to all local energy differences,
  yielding a subset of elements $\mathcal{T}_{\text{D}}$
- sort the marked elements in ascending order (elements with highest energy difference come last)
- loop over all marked elements $T \in \mathcal{T}_{\text{D}}$
  - using a local solver, compute the local increment $d_n^{(T)}$
  - update the local element's degrees of freedom (DOF) using $d_n^{(T)}$
- repeat for $N$ sweeps
