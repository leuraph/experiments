# Summary
This directory cnsiders problems of the form

$$
\begin{align*}
-\nabla(\mathbf{A}(\mathbf x)\nabla u(\mathbf x))
+ \phi (u (\mathbf x))
&=
f(\mathbf x),
\quad
\mathbf x \in \Omega
\\
u(\mathbf x) &= 0,\quad\mathbf x \in \partial \Omega,
\end{align*}
$$

$$
J(u) :=
\frac12 \int_\Omega \nabla u^\top \mathbf A \nabla u ~\mathrm{d}\mathbf x
+\int_\Omega
\underbrace{
      \int_0^{u (\mathbf x)} \phi(\tau)
    ~\mathrm{d} \tau
}_{=: \Phi(u(\mathbf x))}
~\mathrm d \mathbf x -
\int_\Omega f(\mathbf x) u(\mathbf x) ~\mathrm d \mathbf x.
$$

Choosing $(u, v) := \sum_k u_k v_k$ on $\mathbb P_1 (\mathcal T_N)$,
we receive the Riesz representative
$$
DJ(u)_j
=
\sum_i U^i \int_\Omega \nabla \phi_i \mathbf A \nabla \phi_j
+
\int_\Omega \phi( u(\mathbf x) ) \phi_j(\mathbf x) ~\mathrm{d}\mathbf x
-
\int_\Omega f(\mathbf x) \phi_j(\mathbf x).
$$

# Problem 1
Here, we provide
- $\phi(u) := \exp (u) \quad \Rightarrow \quad \Phi(u(\mathbf x)) = \int_0^{u(\mathbf x)} \phi(\tau)~\mathrm d \tau = \exp (u(\mathbf x)) - 1$,
- $\mathbf A (\mathbf x) := \mathbb 1_{2\times 2}$,
- and impose the solution $u(x,y) := x(x-1)y(y-1)$.

In conclusion, this implies the right-hand side
(one may use the script `compute_rhs.py` to compute this function symbolically using `sympy`)
$$
f(x,y)=-2x(x - 1) - 2y(y - 1) + \exp \big[xy(x - 1)(y - 1)\big].
$$