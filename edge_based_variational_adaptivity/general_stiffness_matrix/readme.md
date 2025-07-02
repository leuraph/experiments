# Summary

In this folder, we consider problems of the form
$$
\nabla(A(x) \nabla u(x)) + c u(x) = 1,
$$
where $A(x) \in \mathbb{R}^2$,
$c \in \{0, 1\}$,
$\Omega = (0, 1)^2$,
and we impose homogeneous boundary conditions, i.e.
$u(x) = 0$ for $x \in \partial \Omega$.
Essentially, the problems differ only in the choice of
the matrix $A(x)$.

---

# Problems considered

## Problem 1 (Anisotropic)

$$
A(x) =
\begin{pmatrix}
-1 & 0 \\ 0 & -10^{-2}
\end{pmatrix},
\quad
c=1
$$

## Problem 2 (Singularly Perturbed)

$$
A(x) = - 10^{-2}
\begin{pmatrix}
1 & 0 \\ 0 & 1
\end{pmatrix},
\quad
c=1
$$

## Problem 3 (Piecewise constant)

$$
A(x) = -\kappa (x) \text{Id}_{2\times 2},
\quad
c=1,
\quad
\kappa(x) =
\begin{cases}
1, & x\in \Omega \setminus (\Omega_1 \cup \Omega_2 \cup \Omega_3)
\\
10^2, & x \in \Omega_1 \\
10^4, & x \in \Omega_2 \\
10^6, & x \in \Omega_3 \\
\end{cases},
$$
where
$$
\begin{align*}
\Omega &= (0, 1)^2 \\
\Omega_1 &= (0.1, 0.3) \times (0.1, 0.2) \\
\Omega_2 &= (0.4, 0.7) \times (0.1, 0.3) \\
\Omega_3 &= (0.4, 0.6) \times (0.5, 0.8)
\end{align*}
$$

## Problem 4 (Piecewise constant)

$$
A(x) = -\kappa (x) \text{Id}_{2\times 2},
\quad
c=1,
\quad
\kappa(x) =
\begin{cases}
1, & x\in \Omega \setminus (\Omega_1 \cup \Omega_2 \cup \Omega_3)
\\
10, & x \in \Omega_1 \\
0.1, & x \in \Omega_2 \\
0.05, & x \in \Omega_3 \\
\end{cases},
$$
where
$$
\begin{align*}
\Omega &= (0, 1)^2 \\
\Omega_1 &= (0.1, 0.3) \times (0.1, 0.2) \\
\Omega_2 &= (0.4, 0.7) \times (0.1, 0.3) \\
\Omega_3 &= (0.8, 1.0) \times (0.7, 1.0)
\end{align*}
$$

## Problem 5 (Poisson Equation on L-shape)
This last problem does not _really_ fit into the "general stiffness" matrix setting.
However, as we wish to put this experiment in the very same paper as the experiments above (eva2025), for the sake of convenience and congruency,
we put this experiment in this folder, too.

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


---

# Experiments / Scripts

## Experiment 01 (Problem 1 with adaptive Energy Arioli)
Considers the solution of *Problem 1* by using
CG iterations on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Experiment 02 (Problem 2 with adaptive Energy Arioli)
Considers the solution of *Problem 2* by using
CG iterations on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Experiment 03 (Problem 1 with energy flattening-off)
Considers the solution of *problem 1* by using
CG iterations until
$$
E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}
$$
is met.
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 04 (Problem 2 with energy flattening-off)
Considers the solution of *problem 2* by using
CG iterations until
$$
E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}
$$
is met.
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 05 (Problem 3 with Energy-Arioli)
Considers the solution of *Problem 3* by using
CG iterations on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Experiment 06 (Problem 3 with energy flattening-off)
Considers the solution of *problem 3* by using
CG iterations until
$$
E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}
$$
is met.
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 07 (Problem 3 with Energy-Arioli and precinditioner $P=\text{diag}(\text{lhs})^{-1}$)
Considers the solution of *Problem 3* by using
CG iterations with a diagonal prconditioner on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Experiment 08 (Problem 3 with energy flattening-off and precinditioner $P=\text{diag}(\text{lhs})^{-1}$)
Considers the solution of *problem 3* by using
CG iterations with a diagonal prconditioner until
$$
E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}
$$
is met.
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 09 (Problem 4 with Energy-Arioli and preconditioner $P=\text{diag}(\text{lhs})^{-1}$)
Considers the solution of *Problem 4* by using
CG iterations with a diagonal prconditioner on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Experiment 10 (Problem 4 with energy flattening-off and preconditioner $P=\text{diag}(\text{lhs})^{-1}$)
Considers the solution of *problem 4* by using
CG iterations with a diagonal prconditioner until
$$
E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}
$$
is met.
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 11 (Problem 4 with combined stopping criterion and preconditioner $P=\text{diag}(\text{lhs})^{-1}$)
Considers the solution of *problem 4* by using
CG iterations with a diagonal prconditioner until both
1. Energy Flattening-off indicates convergence
2. Adaptive Energy Arioli indicates convergence
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 12 (Problem 5 with adaptive Energy Arioli)
Considers the solution of *Problem 5* by using
CG iterations on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Experiment 13 (Problem 5 with energy flattening-off)
Considers the solution of *problem 5* by using
CG iterations until
$$
E(u^{n-1}) - E(u^{n}) < \alpha \frac{E(u^{n_{\text{min}}}) - E(u^n)}{n - n_{\text{min}}}
$$
is met.
Then, we perform edge-based variational adaptivity to refine the mesh.

## Energy for problem 01
Generates a reference value of
$\|u\|_a^2$, where $u$ is the exact solution of problem 1 by
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

We ran this script on the cluster, allocating 100G of memory.
The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776, Galerkin solution energy norm squared = 0.07016081941972444
nDOF = 7246, Galerkin solution energy norm squared = 0.0709454930902812
nDOF = 29271, Galerkin solution energy norm squared = 0.07115068182749656
nDOF = 117661, Galerkin solution energy norm squared = 0.0712017943081311
nDOF = 471801, Galerkin solution energy norm squared = 0.07121445506096426
nDOF = 1889521, Galerkin solution energy norm squared = 0.07121759903501006
nDOF = 7562721, Galerkin solution energy norm squared = 0.07121838188085848
```

Our reference value for problem 01 is therefore
$$
\|u\|^2_a \approx \|u^{7562721}_h\|^2_a
=
0.07121838188085848,
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_01.dat`

## Energy for problem 02
Generates a reference value of
$\|u\|_a^2$, where $u$ is the exact solution of problem 2 by
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

We ran this script locally.
The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776, Galerkin solution energy norm squared = 0.6475603281380692
nDOF = 7246, Galerkin solution energy norm squared = 0.6500986689462869
nDOF = 29271, Galerkin solution energy norm squared = 0.6507348578623908
nDOF = 117661, Galerkin solution energy norm squared = 0.6508929268198772
nDOF = 471801, Galerkin solution energy norm squared = 0.650932254841056
nDOF = 1889521, Galerkin solution energy norm squared = 0.65094205873772
nDOF = 7562721, Galerkin solution energy norm squared = 0.6509445059014127
```

Our reference value for problem 02 is therefore
$$
\|u\|^2_a \approx \|u^{7562721}_h\|^2_a
=
0.6509445059014127
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_02.dat`


## Energy for problem 03
Generates a reference value of
$\|u\|_a^2$, where $u$ is the exact solution of problem 3 by
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

We ran this script locally.
The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776, Galerkin solution energy norm squared = 0.026667819476661962
nDOF = 7246, Galerkin solution energy norm squared = 0.027921257728616127
nDOF = 29271, Galerkin solution energy norm squared = 0.028138367251330947
nDOF = 117661, Galerkin solution energy norm squared = 0.028409062970528115
nDOF = 471801, Galerkin solution energy norm squared = 0.028610912906002713
nDOF = 1889521, Galerkin solution energy norm squared = 0.028661547745637728
nDOF = 7562721, Galerkin solution energy norm squared = 0.028670618542377395
```

Our reference value for problem 03 is therefore
$$
\|u\|^2_a \approx \|u^{7562721}_h\|^2_a
=
0.028670618542377395
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_03.dat`


## Energy for problem 04
Generates a reference value of
$\|u\|_a^2$, where $u$ is the exact solution of problem 4 by
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

We ran this script locally.
The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776, Galerkin solution energy norm squared = 0.039422862915199364
nDOF = 7246, Galerkin solution energy norm squared = 0.04003515886758985
nDOF = 29271, Galerkin solution energy norm squared = 0.04043831886269898
nDOF = 117661, Galerkin solution energy norm squared = 0.040636811291174774
nDOF = 471801, Galerkin solution energy norm squared = 0.04068996171899333
nDOF = 1889521, Galerkin solution energy norm squared = 0.040720518610540185
nDOF = 7562721, Galerkin solution energy norm squared = 0.04074476586361528
```
Our reference value for problem 04 is therefore
$$
\|u\|^2_a \approx \|u^h_{7562721}\|^2_a
=
0.04074476586361528
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_04.dat`

## Energy for Problem 5
Note that the analytical solution of this problem remains unknown.
However, we do have an estimate of the value $\| u \|_a^2$ at hand,
i.e. we received, via mail (Patrick Bammer), the value
$$
\| u \|_a^2 \approx 0.214075802220546
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_05.dat`.

# Workflow
1. Perform an experiment
2. calculate errors for the results using the corresponding post-processing script
3. plot the results using the corresponding post-processing script 

# References
- [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.
