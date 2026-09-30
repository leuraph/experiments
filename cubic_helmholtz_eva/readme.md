# Summary

This folder contains experiments for the Helmholtz equation
$$
- \Delta u - \lambda u + u^3 = 0
$$
with homogeneous boundary conditions on the L-shape domain
$$
\Omega_{\text{L}} := (-1, 1)^2 \setminus ([0, 1]\times[-1, 0]).
$$.
Here, we explicitly choose
$$
\lambda \in (\lambda_1, \lambda_2),
$$
i.e., $\lambda$ lies strictly between the first and the second
eigenvalue of the negative Laplacian on the L-shape domain.

Note that we have the following implication
1. $\lambda > \lambda_1$ $\Rightarrow$ Convexity fails $\Rightarrow$ Standard error estimation techniques fail.
2. $\lambda \in (\lambda_1, \lambda_2)$ $\Rightarrow$ We have three solutions $u \in \{0, \pm u^\star\}$, where $\pm u^\star$ are local minima of the energy and both are global minimizers, i.e., $J(\pm u^\star) \leq J(u)$, $\forall u \in H^1_0(\Omega_{\text{L}})$

> Q: I am not sure where to find a proof for the second statement.

From [BT05], we have that the choice $\lambda := 10$ satisfies
$\lambda_1 < \lambda < \lambda_2$.

## References
- [BT05] Reviving the method of particular solutions, Betcke and Trefethen, 2005.