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
  $\Delta E_j = \max \{ \alpha \Delta E^{ \text{local}}_j, \Delta E^{\text{VA}}_j \}$,
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
- perform a collective global update of all local increments
  (local updates are overwritten by local updates corresponding to bigger energy drops)
- mark all elements for refinement for which
  $\alpha \sqrt{\text{dof}} \Delta E^{ \text{local}}_T < \Delta E^{\text{VA}}_T,$
  where $\alpha$ is a fudge parameter, i.e.
  where refining is $\alpha-$times better than solving (energy-wise)
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

## Experiment 9
Line-Search in the direction of the negative gradient, i.e.
$$
\mathbf{x}_{n+1} := \mathbf{x}_n + \frac{\mathbf{r}_n^\top \mathbf{r}_n}{\mathbf{r}_n^\top \mathbf{A} \mathbf{r}_n} \mathbf{r}_n,
$$
alongside ideal convergence criterion, i.e.
$$
\| u_h - u_h^n \| \leq C \text{dof}^{-1/2}.
$$

1. initialize a random solution which is equal to zero on the boundary of the domain.
2. on a given mesh, perform the global update a minimum and maximum number of times where, after each update, we also check for convergence in the above sense. 

## Experiment 10
Line-Search in the direction of the negative gradient, i.e.
$$
\mathbf{x}_{n+1} := \mathbf{x}_n + \frac{\mathbf{r}_n^\top \mathbf{r}_n}{\mathbf{r}_n^\top \mathbf{A} \mathbf{r}_n} \mathbf{r}_n,
$$
alongside energy level-off stopping criterion, i.e.
$$
E(u_h^n) - E(u_h^{n+1}) \leq \alpha \frac{E(u_h^0) - E(u_h^n)}{n},
$$

where $\alpha \in (0,1 )$ is a fudge parameter.
1. initialize a random solution which is equal to zero on the boundary of the domain.
2. on a given mesh, perform the global update a minimum and maximum number of times where, after each update, we also check for convergence in the above sense. 

## Eperiment 11
_Local Solving and local decision on refinement._

In this experiment, we do the following.

- initialize a random solution which is equal to zero on the boundary of the domain.

Then, repeat the following a fixed number of times.
- loop over all alements $T \in \mathcal{T}$.
  - compute the local increment $\Delta x$ and its corresponding energy gain $\Delta E^{\text{local}}$
  - compute the possible energy gain with variational adaptivity, i.e locally red-refine and solving
- perform a collective global update of all local increments
  (local updates are overwritten by local updates corresponding to bigger energy drops)
- define a boolean vector
  $\text{refine}_T := \alpha \Delta E_T^{\text{solve}} < \Delta E_T^{\text{refine}}$
- define a vector
  $\Delta E_T := \max\{ \alpha \Delta E_T^{\text{solve}}, \Delta E_T^{\text{refine}} \}$
- perform a dörfler marking with $\Delta E$, yielding a vector $\text{marked}$.
- redefine $\text{refine} := \text{refine} \, \& \, \text{marked}$
- perform a collective `NV` refinement for all elements marked by $\text{refine}$.

## Experiment 12
Very same as experiment 9 but steepest descent switched with CG

## Experiment 13
Very same as experiment 10 but steepest descent switched with CG

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