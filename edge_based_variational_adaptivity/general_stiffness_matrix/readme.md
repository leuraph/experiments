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

## Problem 1 (Anisotropic)

$$
A(x) =
\begin{pmatrix}
-1 & 0 \\ 0 & -10^{-2}
\end{pmatrix}
$$

## Problem 2 (Singularly Perturbed)

$$
A(x) = - 10^{-1}
\begin{pmatrix}
1 & 0 \\ 0 & 1
\end{pmatrix}
$$

---

# Experiments / Scripts

## Experiment 01
Considers the solution of Problem 1 by using
CG iterations on each mesh with an energy version of
Ariolis stopping criterion [1] in combination with an adaptive
choice of the delay parameter in the HS-estimate.
After convergence, the mesh is refined using edge-based
variational adaptivity with the last iterate, i.e.
we only solve exactly for academic reference and not to refine the mesh.

## Energy for experiment 01
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

## Energy for experiment 02
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

# References
- [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.
