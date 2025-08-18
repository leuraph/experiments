# Summary

This directory holds experiments intended to provide
qualitative insights whether EVA
exhibits an energy-contraction property, i.e.,
we want to numerically investigate whether the
following inequality holds true for a series of examples:
$$
E(u^h_{N_{k+1}}) - E(u) <
q \bigg(E(u^h_{N_k}) - E(u)\bigg),
\quad q\in (0, 1),
$$
where $u^h_{N_k}, u^h_{N_{k+1}}$ are the respective
Galerkin solutions in the spaces
$\mathbb V_{N_k}$ and $\mathbb V_{N_{k+1}}$
and $\mathbb V_{N_{k+1}}$ is obtained by running EVA on the pair
$(u^h_{N_k}, \mathbb V_{N_k})$
and either
-   marking only **one** edge for refinement, i.e.
    the (possibly non-unique) edge corresponding to the biggest
    energy reduction indicator from EVA,
-   or using dörfler marking on the energy reduction indicators
    from EVA.

## Problem 1 (Poisson Equation on L-shape)
This problem is given by the BVP
$$
\begin{align*}
- \Delta u(x) &= 1, \quad x \in \Omega, \\
u(x) &= 0, \quad x \in \partial \Omega,
\end{align*}
$$
where $\Omega := (-1, 1)^2 \setminus [0,1] \times [-1, 0]$,
i.e., an L-shaped domain obtained by removing the fourth
quadrant from the square of side length $2$ and centered at the origin.

## Energy for Problem 1
Note that the analytical solution of this problem remains unknown.
However, we do have an estimate of the value $\| u \|_a^2$ at hand,
i.e. we received, via mail (Patrick Bammer), the value
$$
\| u \|_a^2 \approx 0.214075802220546.
$$
Using the equation
$$
0 = \|u - u\|_a^2 = 2 E(u) + \|u\|_a^2,
$$
we find
$$
E(u) = -\frac{1}{2} \|u\|_a^2.
$$


## Experiment 1 (Problem 1 marking only one edge based on EVA)

This experiment considers the solution of Problem 1.
After an initial refinement,
we solve for the Galerkin solution
and then, using EVA, mark only one edge corresponding
to the biggest energy decay, for refinement.

## Experiment 2 (Problem 1 with Dörfler marking based on EVA)

This experiment considers the solution of Problem 1.
After an initial refinement,
we solve for the Galerkin solution and then, using EVA
together with a prescribed Dörfler parameter,
mark a subset of all edges for refinement.