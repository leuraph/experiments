# SUMMARY

This experiment considers the iterative solution
of the Poisson equation on a fixed mesh, using what we call "local-solvers",
i.e. given an iterate $x_n$, we generate $x_{n+1}$ by considering a single
element $T \in \mathcal{T}$ and solve a simplified local problem on $T$.
The specific scheme differs from solver to solver, see the implementations
of the abstract base class
`iterative_methods.local_solvers.generic_local_solver.GenericLocalSolver`
for details.
Here, we loop over all triangles and consider a local solve on each triangle,
i.e. we simply perform full sweeps.

## directory structures
Running `experiment.py` will drop solutions in files
`results/<solver>/solutions/<n_local_solves>.pkl`,
where `<n_local_solves>` is the total number of local
solve-steps that were used to generate the solution.

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