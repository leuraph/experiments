# Summary

This folder considers numerical experiments for
iterative methods of the form
$$
\Big\langle
    A[u^n_N](u^{n+1} - u^n_N),
    v
\Big\rangle_{\mathbb X' \times \mathbb X}
=
\Delta \tau_n
\Big\langle
    - E'(u^n_N),
    v
\Big\rangle_{\mathbb X' \times \mathbb X}
\quad \forall v \in \mathbb{X}_N,
$$
on a fixed discrete subspace $\mathbb X_N \subset \mathbb X$.

# Problems

## Problem 1

This problem is taken from [HW20].
The PDE is assumed to be given by
$$
-\nabla \cdot \Big(\mu(|\nabla u|^2) \nabla u\Big) = f
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
-\nabla \cdot \Big(\mu(|\nabla u|^2) \nabla u\Big) = f
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

# Theory

## Direct error estimation
We know that we have the estimate
$$
\Big\|u_N^\star - u^{n+1}_N\Big\|_{\mathbb X}
\leq
\left(
    1
    +
    \frac{\beta}{\eta_{\text{m}}\Delta\tau_n}
\right)
\Big\| u^{n+1}_N - u^n_N \Big\|_{\mathbb X},
$$
where
- $\beta$ is the uniform global bound of the operator $A[w]$,
- $\eta_{\text m}$ is the monotonicity constant of $E'$,
- and $\Delta \tau_n$ is the step-stize in the $n$-th iteration.

## Error estimation via splitting
In the splitting approach, we divide the error in the
two contributions
$$
\Big\|u^{n}_N - u^\star_N\Big\|_{\mathbb X}
\leq
\Big\|u^\star_N - u(t_n)\Big\|_{\mathbb X}
+
\Big\|u(t_n) - u^{n}_N\Big\|_{\mathbb X}.
$$

Then, we have
$$
\Big\|
    u_N^\star - u(t)
\Big\|_{\mathbb X}
\leq
\frac{\left\|E'(u^0_N)\right\|_{\mathbb X'}}{\eta_{\text m}}
\text{exp}
\left[
    -\frac{\eta_{\text{c}} \eta_{\text{m}} t}{\beta^2}
\right],
$$
and, further,
$$
\|u^{n+1}_N - u(t_{n+1})\|^2_A
\leq
\frac{\beta L_E^2}{2\eta_{\text{m}}^2} \text{exp}
\left(
    -\frac{\eta_{\text m}}{2}t_{n+1})
\right)
\sum_{k=0}^n
\Big\| u^{k+1}_N - u^k_N \Big\|_{\mathbb X}^2
\left(
    \text{exp}\left(\frac{\eta_{\text{m}}}{2}t_{k+1}\right)
    -
    \text{exp}\left(\frac{\eta_{\text{m}}}{2}t_{k}\right)
\right).
$$

### References
- [HW20] 