# SUMMARY

This collection of experiments considers the iterative solution
of the Poisson equation on a fixed mesh, using what we call "local-solvers",
i.e. given an iterate $x_n$, we generate $x_{n+1}$ by considering a single
element $T \in \mathcal{T}$ and solve a simplified local problem on $T$.
The specific scheme differs from solver to solver, see the implementations
of the abstract base class
`iterative_methods.generic_local_solver.GenericLocalSolver`
for details.

## Problem statement
$$
\begin{align*}
    - \Delta u(x) &= f(x), &\quad& x\in \Omega, \\
     u(x) &= 0, &\quad& x\in \partial \Omega. \\
\end{align*}
$$

Specifically, we consider $\Omega := (0, 1)^2$
and impose the solution to be
$$
u(x, y) = \sin(2 \pi x) \sin(3 \pi y).
$$

# Reproducing results

To reproduce results, follow the steps below.
Note that throughout the steps, only experiments 2 and 3 require the parameter $\theta$ to be passed to.
This can be understood by looking at the details of the experiments below, i.e.
only experiment 2 and 3 make use of Dörfler marking.

0. make sure you are using the **correct python version**: 
  compare `python --version` with `cat .python-version`
  (we recommend using a python version manager such as
  [`pyenv`](https://github.com/pyenv/pyenv))
1. create a **virtual environment**:
  `python -m venv .venv`
2. **source** the virtual environment:
  `source .venv/bin/activate`
3. install the **requirements**:
  `pip install -r requirements.txt`
    > In case you are having troubles installing requirements, e.g. because of lacking
    > permissions to any of my GitHub repositories, please do not hesitate to
    > [contact me](mailto:raphaelleu95@gmail.com).
4. **run an experiment**:
  `python experiment_<N>.py --theta <theta>`
5. calculate the **energy norm errors**:
  `python calculate_energy_norm_errors.py --experiment <N> --theta <theta>`
    > Note that this script calculates the energy norm distance of
    > the iterates $u_n$ to the Galerkin solution $u_h$
    > and not to the exact solution $u$.
6. generate **plots**: We provide a selection of post-processing scripts to generate plots, i.e.
  - for every experiment, plot _energy norm error vs. number of solves_ for all solvers in one plot:
    `python plot_all_errors.py --experiment <N> --theta <theta> -o <path/to/output_file>.pdf`
  - for every solver, plot _energy norm error vs. number of solves_  for all experiments in one plot:
    `python plot_errors_for_all_experiments.py --theta <theta> -o <path/to/output_file>.pdf`
  - for one solver and one experiment, plot the _energy norm error vs. elapsed CPU time_:
    `python plot_energy_norm_error_vs_time.py --path <path_to_dir_holding_solutions_and_elapsed_times_dirs> -o <path/to/output_file>.pdf`

## results directory structures
Running an `experiment_<N>.py` will 
  - drop solutions in files
    `results/<N>/{<theta>/}<solver>/solutions/<n_local_solves>.pkl`,
    where `<n_local_solves>` is the total number of local
    solve-steps that were used to generate the solution
  - drop (accumulated) elapsed CPU time (s) in files
    `results/<N>/{<theta>/}<solver>/elapsed_times/<n_local_solves>.pkl`
  - drop the mesh to `results/<N>/mesh/`
    (expected to be the same for all experiments as we use the same input and same innitial refinement)

# Experiments
In the following, we give a brief description of what is happening
in the different experiments.

## `experiment_1.py`
- sweep over all elements $T \in \mathcal{T}$
  - using a local solver, compute the local increment $d_n^{(T)}$
  - update the local element's degrees of freedom (DOF) using $d_n^{(T)}$
- repeat for $N$ sweeps

## `experiment_2.py`
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

## `experiment_3.py`
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