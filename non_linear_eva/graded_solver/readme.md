## Summary
This folder holds the relevant scripts to generate
graded meshes for the L-shape domain with a singularity
at the origin. Code provided by Florian Spicher.
For an introduction into the topic, see his paper [1].

## NOTES
- the graded meshes are produced with a MATLAB code,
  hence, the indexing starts at `1` and not at `0`.
  This must be taken into account if you load the data
  into `python` code.
- the graded meshes have been generated with
  the MATLAB script `refinement_script.m`.

## References
[1] https://arxiv.org/abs/2504.11292
