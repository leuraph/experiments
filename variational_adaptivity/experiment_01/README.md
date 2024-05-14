# Experiment 01

The experiment is performed as follows.

1.  start with an *initial mesh* `(*)` that has
    undergone several initial uniform refinement steps
    in order to start with a reasonable mesh
2.  perform variational adaptivity and measure the time needed (`dt`)
    for each step
```sh
    x-----x-----x-----x-----x------> [dt]
(x)----x-----x-----x-----x-----x---> [solution]
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

Note that you need to pass a value of `theta` to the script `experiment.py`.
This value indicates the `theta` value used in the Dörfler Marking.
After installing all dependencies in `requirements.txt`, 
performing an experiment is as simple as doing the following on the command line
```bash
python experiment.py --theta 0.5
```