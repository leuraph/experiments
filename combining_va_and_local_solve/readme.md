# Summary

---

# Experiments

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
  - compute the possible energy gain wit¨¨h variational adaptivity, i.e locally red-refine and solving
- put the energy differences in a vector $\Delta E$, where we save
  $\Delta E_j = \max \{ \Delta E^{\alpha \text{local}}_j, \Delta E^{\text{VA}}_j \}$,
  where $\alpha$ is a fudge parameter.
- perform all local updates corresponding to entries in $\Delta E$,
  i.e. for all $T \in \mathcal{T}$,where $\alpha \Delta E^{\text{local}}_j > \Delta E^{\text{VA}}_j$
  such that updates corresponding to biggest local energy drop come last.
- Perform a dörfler marking for all elements marked for refinement
- perform all marked refinements using collective `NV` refinement

## Eperiment 3
In this experiment, we do the following.

1. initialize a random solution which is equal to zero on the boundary of the domain.
2. For all alements $T \in \mathcal{T}$, compute the local increment $\Delta x$
    and its corresponding energy gain $\Delta E^{\text{local}}$.
3. perform all local increments in ascending order (correspodning to local energy gain).
4. start over at point 2 (`sweeps` times)
5. compute the exact solution as reference and dump it as well
6. using the exact solution on current mesh, perform variational adaptivty, i.e. 
  - calculate energy gains for refining,
  - perform dörfler marking,
  - refine the marked elements using NVB and linearly interpolate the current iterate (not the exact solution).
7. start over point 2.

## Experiments 4, 5, 6, 7, 8
In these experiment we want to find out
- if the local solver is capable of ensuring $\| u_h - u_h^n \|_a \propto \text{dof}^{-1/2}$,
- how many sweeps per subspace are needed to ensure it.

To do so, we do the following

1. initialize a random solution which is equal to zero on the boundary of the domain.

2. Compute the exact Galerkin solution as reference and dump it as well
3. For all alements $T \in \mathcal{T}$, compute the local increment $\Delta x$ and local energy gain $\Delta E(T)$.
4. perform all local increments in ascending order (correspodning to local energy gain).
5. check the `stopping criterion` for the experiment at hand:
  - if it is satisfied, perform dörfler markinng using $u_h$, refine, and go to step 2.
  - if it is not satisfied, return to step 3.

### Stopping Criteria
- __Experiment 4__ (control relative energy error):
  $$
  \frac{\| u_h - u_h^n \|_a}{\| u_h \|_a }
  < \text{tol}$$
- __Experiment 5__ (control relative change in energy norm):
  $$
  \frac{\| u_h^n - u_h^{n+1} \|_a}{\| u_h^n \|_a} < \text{tol}$$
- __Experiment 6__ (control relative change in energy): 
  $$
  \frac{\| E(u_h^n) - E(u_h^{n+1}) \|_a}{\| E(u_h^n) \|_a} < \text{tol}$$
- __Experiment 7__ (force ideal convergence slope of $-1$): 
  - On $\mathbb{V}_0$, perform 50 ierations.
  - On $\mathbb{V}_N$, check for ideal convergence rate via slope in log log plot, i.e. compute
    $$
    \frac{
      \log \| u - u^n_{N+1} \|_a^2 
      - \log \| u - u^\star_N \|_a^2
    }{
      \log \text{dof}(\mathbb{V}_{N+1}) 
      - \log \text{dof}(\mathbb{V}_N)
    } < -1 + \text{tol}
    $$
- __Experiment 8__ (force ideal convergence scaling): 
  $$\| u_h - u_h^n \|_a \leq c \, \text{dof}^{-1/2}$$

---

# Scripts

## `plot_energies_for_sweeps.py`
Plots the energy decay per sweep on a given mesh, i.e. $E(u_N^n)$ vs. $n$.

## `plot_number_of_iterations_for_exp.py`
Plots the number of iterations (sweeps) on a given mesh, i.e. if $n_N^*$ is the number iterations such that $u_N^{n^*_N} = u^*_N$,
then this script plots $n_N^*$ against $n_{\text{dof}}(\mathbb{V}_N)$.

## `plot_errors_for_exp.py`
Plots the energy error for each iterand $u^n_N$, i.e.
$\| u - u_N^n \|_a^2$
against $n_{\text{dof}}(\mathbb{V}_N)$