# Summary

This folder considers numerical experiments for
iterative methods of the form
$$
\Big\langle
    A[u^n_N](u^{n+1} - u^n_N),
    v
\Big\rangle_{\mathbb X' \times \mathbb X}
=
\Big\langle
    - E'(u^n_N),
    v
\Big\rangle_{\mathbb X' \times \mathbb X}
\quad \forall v \in \mathbb{X}_N.
$$

## Problem 1

This problem is taken from [HW20].
The PDE is assumed to be given by
$$
-\nabla \Big(\mu(|\nabla u|^2) \nabla u\Big) = f
$$
on the L-shaped domain.
The nonlinearity $\mu$ is take to be
$$
\mu (t) := 1 + e^{-t}.
$$
Then, we have
$$
m_\mu (t-s) \leq \mu(t^2)t - \mu(s^2)s \leq M_\mu (t-2),
\quad
t\geq s \geq 0,
$$
with the constants
- $m_\mu = 1-2e^{-3/2}$,
- $M_\mu = 2$.

## Problem 2
This problem is taken from [HW20].
The PDE is assumed to be given by
$$
-\nabla \Big(\mu(|\nabla u|^2) \nabla u\Big) = f
$$
on the L-shaped domain.
The nonlinearity $\mu$ is take to be
$$
\mu (t) := (1 + t)^{-1} + \frac12.
$$
Then, we have
$$
m_\mu (t-s) \leq \mu(t^2)t - \mu(s^2)s \leq M_\mu (t-2),
\quad
t\geq s \geq 0,
$$
with the constants
- $m_\mu = 3/8$,
- $M_\mu = 3/2$.

### References
- [HW20] 