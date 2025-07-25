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

## Experiment 14 (Problem 1 with combined stopping criterion)
Considers the solution of *problem 1* by using
CG iterations until both
1. Energy Flattening-off indicates convergence
2. Adaptive Energy Arioli indicates convergence
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 15 (Problem 2 with combined stopping criterion)
Considers the solution of *problem 2* by using
CG iterations until both
1. Energy Flattening-off indicates convergence
2. Adaptive Energy Arioli indicates convergence
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 16 (Problem 4 with combined stopping criterion and preconditioner $P=\text{diag}(\text{lhs})^{-1}$)
Considers the solution of *problem 4* by using
CG iterations with a diagonal prconditioner until both
1. Energy Flattening-off indicates convergence
2. Adaptive Energy Arioli indicates convergence
Then, we perform edge-based variational adaptivity to refine the mesh.

## Experiment 17 (Problem 5 with combined stopping criterion)
Considers the solution of *problem 5* by using
CG iterations until both
1. Energy Flattening-off indicates convergence
2. Adaptive Energy Arioli indicates convergence
Then, we perform edge-based variational adaptivity to refine the mesh.

# Reference Values of Energy norm of solutions
As the exact solutions of all problems remains unknown,
we can only provide reference values of the energy norm thereof, i.e.
we strive to approximate the value of $\|u\|_a^2$.
This is done in different ways.
For problems 1,2, and 4, we compute reference values
by uniformly refining the mesh and
1. solve for the Galerkin solution,
2. approximate the Galerkin solution by using CG with a predescribed relative tolerance.
The computations of the latter (CG) were performed using the script
`calculate_reference_value_energy_norm_squared_cg.py`.
We found that solving for the Galerkin solution becomes impractical for meshes
with more than ~7 million degrees of freedom, whereas CG only fails for meshes
with more than ~30 million degrees of freedom.
Both fail because they run out of memory (OOM).

## Problem 01
Reference value for the energy norm squared of the solution to problem 1.

### Direct Solver
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

We ran this script on the cluster, allocating 100G of memory.
The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776,    Galerkin solution energy norm squared = 0.07016081941972444
nDOF = 7246,    Galerkin solution energy norm squared = 0.0709454930902812
nDOF = 29271,   Galerkin solution energy norm squared = 0.07115068182749656
nDOF = 117661,  Galerkin solution energy norm squared = 0.0712017943081311
nDOF = 471801,  Galerkin solution energy norm squared = 0.07121445506096426
nDOF = 1889521, Galerkin solution energy norm squared = 0.07121759903501006
nDOF = 7562721, Galerkin solution energy norm squared = 0.07121838188085848
```
### Conjugate Gradient
1. uniformly refine all elements using NVB,
2. run CG for $Ax=b$ until relative tolerance is reached,
3. and calculate $x^\top A x = \|x\|_a^2$.

The outputs look like this.

```bash
problem number = 1
relative tolerance for CG = 0.01

nDOF = 1776,     converged CG approximation energy norm squared = 0.07016050539198732, n_iterations = 145
nDOF = 7246,     converged CG approximation energy norm squared = 0.07094527552968677, n_iterations = 302
nDOF = 29271,    converged CG approximation energy norm squared = 0.07115057175129276, n_iterations = 637
nDOF = 117661,   converged CG approximation energy norm squared = 0.07120173780802998, n_iterations = 1337
nDOF = 471801,   converged CG approximation energy norm squared = 0.07121442380857763, n_iterations = 2782
nDOF = 1889521,  converged CG approximation energy norm squared = 0.07121758609777623, n_iterations = 5858
nDOF = 7562721,  converged CG approximation energy norm squared = 0.07121837548132107, n_iterations = 12190
nDOF = 30260161, converged CG approximation energy norm squared = 0.07121857362888953, n_iterations = 25152
/var/spool/slurmd.spool/job27752251/slurm_script: line 25: 3346084 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 1 --rtol 1e-2
slurmstepd: error: Detected 1 oom_kill event in StepId=27752251.batch. Some of the step tasks have been OOM Killed.
Calculating CG approximations and their energy norm squared...

problem number = 1
relative tolerance for CG = 0.001

nDOF = 1776,     converged CG approximation energy norm squared = 0.07016081568379504, n_iterations = 192
nDOF = 7246,     converged CG approximation energy norm squared = 0.07094549062873509, n_iterations = 400
nDOF = 29271,    converged CG approximation energy norm squared = 0.07115068075739592, n_iterations = 844
nDOF = 117661,   converged CG approximation energy norm squared = 0.07120179385725417, n_iterations = 1751
nDOF = 471801,   converged CG approximation energy norm squared = 0.07121445476273157, n_iterations = 3567
nDOF = 1889521,  converged CG approximation energy norm squared = 0.07121759889262616, n_iterations = 7421
nDOF = 7562721,  converged CG approximation energy norm squared = 0.07121838181588147, n_iterations = 15338
nDOF = 30260161, converged CG approximation energy norm squared = 0.07121857715538998, n_iterations = 31621
/var/spool/slurmd.spool/job27752254/slurm_script: line 25: 2403191 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 1 --rtol 1e-3
slurmstepd: error: Detected 1 oom_kill event in StepId=27752254.batch. Some of the step tasks have been OOM Killed.
Calculating CG approximations and their energy norm squared...

problem number = 1
relative tolerance for CG = 0.0001

nDOF = 1776,     converged CG approximation energy norm squared = 0.0701608195379945,  n_iterations = 243
nDOF = 7246,     converged CG approximation energy norm squared = 0.07094549304675296, n_iterations = 504
nDOF = 29271,    converged CG approximation energy norm squared = 0.07115068181189303, n_iterations = 1043
nDOF = 117661,   converged CG approximation energy norm squared = 0.07120179430171263, n_iterations = 2142
nDOF = 471801,   converged CG approximation energy norm squared = 0.07121445505758776, n_iterations = 4416
nDOF = 1889521,  converged CG approximation energy norm squared = 0.07121759903475282, n_iterations = 9127
nDOF = 7562721,  converged CG approximation energy norm squared = 0.07121838188538238, n_iterations = 18842
nDOF = 30260161, converged CG approximation energy norm squared = 0.07121857719147966, n_iterations = 38946
/var/spool/slurmd.spool/job27752255/slurm_script: line 25: 4007278 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 1 --rtol 1e-4
slurmstepd: error: Detected 1 oom_kill event in StepId=27752255.batch. Some of the step tasks have been OOM Killed.
Calculating CG approximations and their energy norm squared...

problem number = 1
relative tolerance for CG = 1e-05

nDOF = 1776,     converged CG approximation energy norm squared = 0.07016081941974212, n_iterations = 294
nDOF = 7246,     converged CG approximation energy norm squared = 0.07094549308963909, n_iterations = 615
nDOF = 29271,    converged CG approximation energy norm squared = 0.07115068182743023, n_iterations = 1263
nDOF = 117661,   converged CG approximation energy norm squared = 0.0712017943081612,  n_iterations = 2594
nDOF = 471801,   converged CG approximation energy norm squared = 0.07121445506103177, n_iterations = 5333
nDOF = 1889521,  converged CG approximation energy norm squared = 0.07121759903633725, n_iterations = 10852
nDOF = 7562721,  converged CG approximation energy norm squared = 0.07121838188616177, n_iterations = 22058
nDOF = 30260161, converged CG approximation energy norm squared = 0.07121857719182778, n_iterations = 45076
/var/spool/slurmd.spool/job27752257/slurm_script: line 25: 1448807 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 1 --rtol 1e-5
slurmstepd: error: Detected 1 oom_kill event in StepId=27752257.batch. Some of the step tasks have been OOM Killed.
Calculating CG approximations and their energy norm squared...
```

Our reference value for problem 01 is therefore
$$
\|u\|^2_a \approx \|u^{7562721}_h\|^2_a
=
0.07121838188085848,
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_01.dat`

## Problem 02
Reference value for the energy norm squared of the solution to problem 2.

### Direct Solver
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776,    Galerkin solution energy norm squared = 0.6475603281380692
nDOF = 7246,    Galerkin solution energy norm squared = 0.6500986689462869
nDOF = 29271,   Galerkin solution energy norm squared = 0.6507348578623908
nDOF = 117661,  Galerkin solution energy norm squared = 0.6508929268198772
nDOF = 471801,  Galerkin solution energy norm squared = 0.650932254841056
nDOF = 1889521, Galerkin solution energy norm squared = 0.65094205873772
nDOF = 7562721, Galerkin solution energy norm squared = 0.6509445059014127
```

### Conjugate Gradient
1. uniformly refine all elements using NVB,
2. run CG for $Ax=b$ until relative tolerance is reached,
3. and calculate $x^\top A x = \|x\|_a^2$.

The outputs look like this.

```bash
problem number = 2
relative tolerance for CG = 0.01

nDOF = 1776,     converged CG approximation energy norm squared = 0.6475520820047701, n_iterations = 47
nDOF = 7246,     converged CG approximation energy norm squared = 0.6500944541747053, n_iterations = 99
nDOF = 29271,    converged CG approximation energy norm squared = 0.6507324146227458, n_iterations = 206
nDOF = 117661,   converged CG approximation energy norm squared = 0.6508915643641714, n_iterations = 429
nDOF = 471801,   converged CG approximation energy norm squared = 0.650931481266865,  n_iterations = 896
nDOF = 1889521,  converged CG approximation energy norm squared = 0.650941661239565,  n_iterations = 1886
nDOF = 7562721,  converged CG approximation energy norm squared = 0.6509443073897536, n_iterations = 3968
nDOF = 30260161, converged CG approximation energy norm squared = 0.650945013761668,  n_iterations = 8306
/var/spool/slurmd.spool/job27752258/slurm_script: line 25: 3458401 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 2 --rtol 1e-2
slurmstepd: error: Detected 1 oom_kill event in StepId=27752258.batch. Some of the step tasks have been OOM Killed.

problem number = 2
relative tolerance for CG = 0.001

nDOF = 1776,     converged CG approximation energy norm squared = 0.6475602265947248, n_iterations = 65
nDOF = 7246,     converged CG approximation energy norm squared = 0.6500986089361209, n_iterations = 135
nDOF = 29271,    converged CG approximation energy norm squared = 0.6507348285046098, n_iterations = 282
nDOF = 117661,   converged CG approximation energy norm squared = 0.6508929117923998, n_iterations = 587
nDOF = 471801,   converged CG approximation energy norm squared = 0.6509322481213227, n_iterations = 1230
nDOF = 1889521,  converged CG approximation energy norm squared = 0.6509420559840645, n_iterations = 2550
nDOF = 7562721,  converged CG approximation energy norm squared = 0.6509445044660431, n_iterations = 5232
nDOF = 30260161, converged CG approximation energy norm squared = 0.6509451164616353, n_iterations = 10752
/var/spool/slurmd.spool/job27752259/slurm_script: line 25: 1846648 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 2 --rtol 1e-3
slurmstepd: error: Detected 1 oom_kill event in StepId=27752259.batch. Some of the step tasks have been OOM Killed.

problem number = 2
relative tolerance for CG = 0.0001

nDOF = 1776,     converged CG approximation energy norm squared = 0.6475603274579393, n_iterations = 84
nDOF = 7246,     converged CG approximation energy norm squared = 0.6500986685612253, n_iterations = 172
nDOF = 29271,    converged CG approximation energy norm squared = 0.6507348576582532, n_iterations = 352
nDOF = 117661,   converged CG approximation energy norm squared = 0.6508929267196378, n_iterations = 721
nDOF = 471801,   converged CG approximation energy norm squared = 0.6509322547900139, n_iterations = 1473
nDOF = 1889521,  converged CG approximation energy norm squared = 0.6509420587113476, n_iterations = 3002
nDOF = 7562721,  converged CG approximation energy norm squared = 0.6509445059035608, n_iterations = 6136
nDOF = 30260161, converged CG approximation energy norm squared = 0.6509451171794212, n_iterations = 12622
/var/spool/slurmd.spool/job27752260/slurm_script: line 25: 1153213 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 2 --rtol 1e-4
slurmstepd: error: Detected 1 oom_kill event in StepId=27752260.batch. Some of the step tasks have been OOM Killed.

problem number = 2
relative tolerance for CG = 1e-05

nDOF = 1776,     converged CG approximation energy norm squared = 0.6475603281293694, n_iterations = 98
nDOF = 7246,     converged CG approximation energy norm squared = 0.650098668942585,  n_iterations = 202
nDOF = 29271,    converged CG approximation energy norm squared = 0.6507348578603254, n_iterations = 411
nDOF = 117661,   converged CG approximation energy norm squared = 0.6508929268187976, n_iterations = 838
nDOF = 471801,   converged CG approximation energy norm squared = 0.6509322548409675, n_iterations = 1707
nDOF = 1889521,  converged CG approximation energy norm squared = 0.6509420587403482, n_iterations = 3473
nDOF = 7562721,  converged CG approximation energy norm squared = 0.6509445059199492, n_iterations = 7125
nDOF = 30260161, converged CG approximation energy norm squared = 0.6509451171871544, n_iterations = 14539
/var/spool/slurmd.spool/job27752261/slurm_script: line 25: 57061 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 2 --rtol 1e-5
slurmstepd: error: Detected 1 oom_kill event in StepId=27752261.batch. Some of the step tasks have been OOM Killed.
```

Our reference value for problem 02 is therefore
$$
\|u\|^2_a \approx \|u^{7562721}_h\|^2_a
=
0.6509445059014127
$$
which can be found in the file
`energy_norm_squared_reference_value_problem_02.dat`


## Problem 04
Reference value for the energy norm squared of the solution to problem 4.

### Direct Solver
1. uniformly refine all elements using NVB,
2. solve $Ax=b$ exactly,
3. and calculate $x^\top A x = \|x\|_a^2$.

The highest n_DOF we reached was 7'562'721.
The output of the experiment looks like this.
```sh
nDOF = 1776,    Galerkin solution energy norm squared = 0.039422862915199364
nDOF = 7246,    Galerkin solution energy norm squared = 0.04003515886758985
nDOF = 29271,   Galerkin solution energy norm squared = 0.04043831886269898
nDOF = 117661,  Galerkin solution energy norm squared = 0.040636811291174774
nDOF = 471801,  Galerkin solution energy norm squared = 0.04068996171899333
nDOF = 1889521, Galerkin solution energy norm squared = 0.040720518610540185
nDOF = 7562721, Galerkin solution energy norm squared = 0.04074476586361528
```

### Conjugate Gradient
1. uniformly refine all elements using NVB,
2. run CG for $Ax=b$ until relative tolerance is reached,
3. and calculate $x^\top A x = \|x\|_a^2$.

The outputs look like this.

```bash
problem number = 4
relative tolerance for CG = 0.01

nDOF = 1776,     converged CG approximation energy norm squared = 0.039422725739309716, n_iterations = 109
nDOF = 7246,     converged CG approximation energy norm squared = 0.04003509142917638,  n_iterations = 230
nDOF = 29271,    converged CG approximation energy norm squared = 0.04043828160637063,  n_iterations = 473
nDOF = 117661,   converged CG approximation energy norm squared = 0.040636791882217485, n_iterations = 972
nDOF = 471801,   converged CG approximation energy norm squared = 0.04068995744785822,  n_iterations = 2051
nDOF = 1889521,  converged CG approximation energy norm squared = 0.040720515409082336, n_iterations = 4148
nDOF = 7562721,  converged CG approximation energy norm squared = 0.04074476424872379,  n_iterations = 8467
nDOF = 30260161, converged CG approximation energy norm squared = 0.04075631158690203,  n_iterations = 17155
/var/spool/slurmd.spool/job27752262/slurm_script: line 25: 4171355 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 4 --rtol 1e-2
slurmstepd: error: Detected 1 oom_kill event in StepId=27752262.batch. Some of the step tasks have been OOM Killed.

problem number = 4
relative tolerance for CG = 0.001

nDOF = 1776,     converged CG approximation energy norm squared = 0.03942286265887802, n_iterations = 131
nDOF = 7246,     converged CG approximation energy norm squared = 0.04003515816920987, n_iterations = 270
nDOF = 29271,    converged CG approximation energy norm squared = 0.04043831862402136, n_iterations = 554
nDOF = 117661,   converged CG approximation energy norm squared = 0.04063681115855645, n_iterations = 1126
nDOF = 471801,   converged CG approximation energy norm squared = 0.040689961677195,   n_iterations = 2329
nDOF = 1889521,  converged CG approximation energy norm squared = 0.04072051858373632, n_iterations = 4716
nDOF = 7562721,  converged CG approximation energy norm squared = 0.04074476586398393, n_iterations = 9565
nDOF = 30260161, converged CG approximation energy norm squared = 0.04075631249679278, n_iterations = 19410
/var/spool/slurmd.spool/job27752277/slurm_script: line 25: 302478 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 4 --rtol 1e-3
slurmstepd: error: Detected 1 oom_kill event in StepId=27752277.batch. Some of the step tasks have been OOM Killed.

problem number = 4
relative tolerance for CG = 0.0001

nDOF = 1776,     converged CG approximation energy norm squared = 0.03942286290591484,  n_iterations = 146
nDOF = 7246,     converged CG approximation energy norm squared = 0.04003515888647288,  n_iterations = 304
nDOF = 29271,    converged CG approximation energy norm squared = 0.040438318858596584, n_iterations = 621
nDOF = 117661,   converged CG approximation energy norm squared = 0.040636811289673884, n_iterations = 1267
nDOF = 471801,   converged CG approximation energy norm squared = 0.04068996171869217,  n_iterations = 2606
nDOF = 1889521,  converged CG approximation energy norm squared = 0.04072051861277673,  n_iterations = 5343
nDOF = 7562721,  converged CG approximation energy norm squared = 0.040744765881266555, n_iterations = 10821
nDOF = 30260161, converged CG approximation energy norm squared = 0.04075631250568247,  n_iterations = 21798
/var/spool/slurmd.spool/job27752279/slurm_script: line 25: 4104463 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 4 --rtol 1e-4
slurmstepd: error: Detected 1 oom_kill event in StepId=27752279.batch. Some of the step tasks have been OOM Killed.

problem number = 4
relative tolerance for CG = 1e-05

nDOF = 1776,     converged CG approximation energy norm squared = 0.039422862913769695, n_iterations = 164
nDOF = 7246,     converged CG approximation energy norm squared = 0.04003515886670348,  n_iterations = 340
nDOF = 29271,    converged CG approximation energy norm squared = 0.04043831886269658,  n_iterations = 696
nDOF = 117661,   converged CG approximation energy norm squared = 0.04063681129116415,  n_iterations = 1403
nDOF = 471801,   converged CG approximation energy norm squared = 0.040689961719291526, n_iterations = 2857
nDOF = 1889521,  converged CG approximation energy norm squared = 0.04072051861301904,  n_iterations = 5780
nDOF = 7562721,  converged CG approximation energy norm squared = 0.04074476588139947,  n_iterations = 11673
nDOF = 30260161, converged CG approximation energy norm squared = 0.040756312505732355, n_iterations = 23645
/var/spool/slurmd.spool/job27752280/slurm_script: line 25: 43020 Killed                  python -u calculate_reference_value_energy_norm_squared_cg.py --problem 4 --rtol 1e-5
slurmstepd: error: Detected 1 oom_kill event in StepId=27752280.batch. Some of the step tasks have been OOM Killed.
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
