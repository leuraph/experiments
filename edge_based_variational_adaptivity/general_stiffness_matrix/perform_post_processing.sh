#!/bin/bash

# error calculations
# ------------------
# bash calculate_all_errors.sh results/experiment_01 1 energy_norm_squared_reference_value_problem_01.dat
# bash calculate_all_errors.sh results/experiment_02 2 energy_norm_squared_reference_value_problem_02.dat

# bash calculate_all_errors.sh results/experiment_03 1 energy_norm_squared_reference_value_problem_01.dat
# bash calculate_all_errors.sh results/experiment_04 2 energy_norm_squared_reference_value_problem_02.dat

# bash calculate_all_errors.sh results/experiment_09 4 energy_norm_squared_reference_value_problem_04.dat
# bash calculate_all_errors.sh results/experiment_10 4 energy_norm_squared_reference_value_problem_04.dat

# bash calculate_all_errors.sh results/experiment_12 5 energy_norm_squared_reference_value_problem_05.dat
# bash calculate_all_errors.sh results/experiment_13 5 energy_norm_squared_reference_value_problem_05.dat

# Cobined Stopping Criteria
# bash calculate_all_errors.sh results/experiment_14 1 energy_norm_squared_reference_value_problem_01.dat
# bash calculate_all_errors.sh results/experiment_15 2 energy_norm_squared_reference_value_problem_02.dat
# bash calculate_all_errors.sh results/experiment_16 4 energy_norm_squared_reference_value_problem_04.dat
# bash calculate_all_errors.sh results/experiment_17 5 energy_norm_squared_reference_value_problem_05.dat

# plotting
# --------
bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_01
bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_02

bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_03
bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_04

bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_09
bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_10

bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_12
bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_13

# Combined Stopping Criteria
# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_14
# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_15
# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_16
# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_17