#!/usr/bin/bash

# This script computes all the reference energies,
# adjust the parameters as needed.

python compute_reference_energy.py \
    --problem 1 \
    --reference-solution reference_solutions/problem-1_hmax-0.0003_alpha-0.9_gamma-10.0.pkl \
    --reference-mesh graded_meshes/refined_meshes/hmax-0.0003_n_vertices-9847807 \

python compute_reference_energy.py \
    --problem 2 \
    --reference-solution reference_solutions/problem-2_hmax-0.0003_alpha-0.9_gamma-10.0.pkl \
    --reference-mesh graded_meshes/refined_meshes/hmax-0.0003_n_vertices-9847807 \

python compute_reference_energy.py \
    --problem 3 \
    --reference-solution reference_solutions/problem-3_hmax-0.0003_alpha-0.9_gamma-10.0.pkl \
    --reference-mesh graded_meshes/refined_meshes/hmax-0.0003_n_vertices-9847807 \