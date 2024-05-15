# Summary

This experiment considers the problem

$$
\begin{alignat*}{2}
- \Delta u (x, y) &= f(x, y), & \quad & (x, y) \in (0, 1)^2, \\
u(x, y) &= 0, & \quad & (x, y) \in \partial (0, 1)^2,
\end{alignat*}
$$

where the function $u(x, y)$ is imposed to be

$$
u(x, y)
=
x(x-1)y(y-1)
\exp\left[ -\sigma_x (x-\mu_x)^2 -\sigma_y (y-\mu_y)^2 \right].
$$

The function $f(x, y)$ is defined accordingly
(its symbolic computation is outsourced to the jupyter notebook
`variational_adaptivity/gauss_kick/experiment_notes.ipynb`).

## Experiment's schedule
Each experiment is performed as follows.

1.  start with an *initial mesh* `(*)` that has
    undergone several initial uniform refinement steps
    in order to start with a reasonable mesh
    and initialize `(0)` the first `dt` as zero
    (this step needs to be done to ensure same shape
    for `dt` and `solution` arrays)
2.  perform variational adaptivity and measure the time needed (`dt`)
    for each step
```sh
(0)----x-----x-----x-----x-----x------> [dt]
---(x)----x-----x-----x-----x-----x---> [solution]
```

Note that, in the above outline of the experiment's schedule,
each `*` represents a measurement or checkpoint,
i.e. we measure the time needed (`dt`) to
- calculate all local energy gains,
- refine (NVB) the mesh based on Dörfler marking,
- solve the linear system of equations exactly,

and we save intermediate results (`solution`), i.e.
- mesh (elements and coordinates),
- solution $u_h$,
- energy $a(u_h, u_h)$.

## perform an experiment

Note that you need to pass a value of `theta` to any of the experiment scripts.
This value indicates the `theta` value used in the Dörfler Marking.
After installing all dependencies in `requirements.txt`,
performing an experiment is as simple as doing the following on the command line
```bash
python experiment_<...>.py --theta 0.5
```
The results can then be found in `results/experiment_XY/theta_<theta>/`.

# Experiment 01 (Gauss kick: initial order 1)

Notes
-----
Looking at the results, we notice the refined mesh to be non-symmetric, e.g.
```sh
python plot_mesh.py results/theta_0.6/5
```

# Experiment 02 (Gauss kick: initial order 2)
This experiment considers the same problem as in Experiment 01.
The only difference is the order nodes in `data/elements.dat`, i.e.
the ordering of the edges in the initial mesh.

Notes
-----

# Experiment 03 (Gauss kick: shuffle)
This experiment considers the same problem as in Experiment 01
and uses the same initial mesh as Experiment 02.
However, here we shuffle the element ordering after each refinement to
investigate the effect of element ordering on the symmetry
of the resulting refined mesh.

Notes
-----

# Experiment 04 (Gauss kick: jiggle)
This experiment considers the same problem as in Experiment 01
and uses the same initial mesh as Experiment 02.
However, here we jiggle the non-boundary coordinates
of the initial mesh investigate the effect of the structuredness
of the initial mesh on the symmetry
of the resulting refined mesh.

Notes
-----

# Experiment 05 (Gauss kick: shuffle + jiggle)
This experiment considers the same problem as in Experiment 01
and uses the same initial mesh as Experiment 02.
However, here we jiggle the non-boundary coordinates
of the initial mesh investigate the effect of the structuredness
of the initial mesh on the symmetry
of the resulting refined mesh.

Notes
-----