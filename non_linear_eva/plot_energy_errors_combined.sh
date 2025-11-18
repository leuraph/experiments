#!/usr/bin/bash

# This script generates three plots.
# Each of the plots shows the energy difference vs. number of degrees of freedom
# for the three different stopping criteria.
# Each plot corresponds to one specific problem.

# PROBLEM 1
# ---------
python plot_energy_errors_combined.py \
    --reference-energy reference_energies/problem-1_hmax-0.0003_alpha-0.9_gamma-10.0.pkl \
    --problem 1 \
    --default-path results/problem-1_default_theta-0.5_eta-1.0_miniter-10_gtol-1e-08 \
    --arioli-path results/problem-1_relative-energy-decay_theta-0.5_eta-1.0_fudge-0.1_miniter-10_tau-1.01_initial_delay-10_delay_increase-5 \
    --tail-off-path results/problem-1_energy-tail-off_theta-0.5_eta-1.0_fudge-0.01_miniter-10_batchsize-5

# PROBLEM 2
# ---------
python plot_energy_errors_combined.py \
    --reference-energy reference_energies/problem-2_hmax-0.0003_alpha-0.9_gamma-10.0.pkl \
    --problem 2 \
    --default-path results/problem-2_default_theta-0.5_eta-1.0_miniter-10_gtol-1e-08 \
    --arioli-path results/problem-2_relative-energy-decay_theta-0.5_eta-1.0_fudge-0.1_miniter-10_tau-1.01_initial_delay-10_delay_increase-5 \
    --tail-off-path results/problem-2_energy-tail-off_theta-0.5_eta-1.0_fudge-0.01_miniter-10_batchsize-5

# PROBLEM 3
# ---------
python plot_energy_errors_combined.py \
    --reference-energy reference_energies/problem-3_hmax-0.0003_alpha-0.9_gamma-10.0.pkl \
    --problem 3 \
    --default-path results/problem-3_default_theta-0.5_eta-1.0_miniter-10_gtol-1e-08 \
    --arioli-path results/problem-3_relative-energy-decay_theta-0.5_eta-1.0_fudge-0.1_miniter-10_tau-1.01_initial_delay-10_delay_increase-5 \
    --tail-off-path results/problem-3_energy-tail-off_theta-0.5_eta-1.0_fudge-0.01_miniter-10_batchsize-5