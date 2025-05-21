#!/bin/bash

# error calculations
# ------------------
# bash calculate_all_errors.sh results/experiment_01 1 energy_norm_squared_reference_value_problem_01.dat
# bash calculate_all_errors.sh results/experiment_02 2 energy_norm_squared_reference_value_problem_02.dat

# bash calculate_all_errors.sh results/experiment_03 1 energy_norm_squared_reference_value_problem_01.dat
# bash calculate_all_errors.sh results/experiment_04 2 energy_norm_squared_reference_value_problem_02.dat

# bash calculate_all_errors.sh results/experiment_05 3 energy_norm_squared_reference_value_problem_03.dat
# bash calculate_all_errors.sh results/experiment_06 3 energy_norm_squared_reference_value_problem_03.dat

bash calculate_all_errors.sh results/experiment_07 3 energy_norm_squared_reference_value_problem_03.dat
bash calculate_all_errors.sh results/experiment_08 3 energy_norm_squared_reference_value_problem_03.dat

# plotting
# --------
# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_01
# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_02

# bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_03
# bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_04

# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_05
# bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_06

# bash plot_all_errors.sh plot_energy_errors_arioli.py results/experiment_07
# bash plot_all_errors.sh plot_energy_errors_energy_decay.py results/experiment_08