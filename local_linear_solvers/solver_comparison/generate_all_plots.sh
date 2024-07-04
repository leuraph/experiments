
thetas=(0.2 0.4 0.5 0.6 0.8)

for theta in "${thetas[@]}"; do
    # Plotting results for experiment 1, 2, and 3
    # -------------------------------------------
    output_file="plots/energy_norm_errors_experiment-1.pdf"
    MPLBACKEND=Agg python plot_all_errors.py --experiment 1 -o "${output_file}"

    output_file="plots/energy_norm_errors_experiment-2_theta-${theta}.pdf"
    MPLBACKEND=Agg python plot_all_errors.py --experiment 2 --theta "${theta}" -o "${output_file}"

    output_file="plots/energy_norm_errors_experiment-3_theta-${theta}.pdf"
    MPLBACKEND=Agg python plot_all_errors.py --experiment 3 --theta "${theta}" -o "${output_file}"

    # for each solver, plot all experiments in one plot
    # -------------------------------------------------
    MPLBACKEND=Agg python plot_errors_for_all_experiments.py --theta "${theta}" -o "plots"

    # plot all error vs. CPU time plots
    # ---------------------------------
    # local jacobi
    output_file="plots/CPU_times/error_vs_CPU_time_local_jacobi_experiment-1.pdf"
    input_dir="results/1/local_jacobi/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_jacobi_experiment-2_theta-${theta}.pdf"
    input_dir="results/2/${theta}/local_jacobi/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_jacobi_experiment-3_theta-${theta}.pdf"
    input_dir="results/3/${theta}/local_jacobi/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    # local block jacobi
    output_file="plots/CPU_times/error_vs_CPU_time_local_block_jacobi_experiment-1.pdf"
    input_dir="results/1/local_block_jacobi/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_block_jacobi_experiment-2_theta-${theta}.pdf"
    input_dir="results/2/${theta}/local_block_jacobi/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_block_jacobi_experiment-3_theta-${theta}.pdf"
    input_dir="results/3/${theta}/local_block_jacobi/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    # local context solver simultaneous
    output_file="plots/CPU_times/error_vs_CPU_time_local_context_simult_experiment-1.pdf"
    input_dir="results/1/local_context_solver_simultaneous/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_context_simult_experiment-2_theta-${theta}.pdf"
    input_dir="results/2/${theta}/local_context_solver_simultaneous/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_context_simult_experiment-3_theta-${theta}.pdf"
    input_dir="results/3/${theta}/local_context_solver_simultaneous/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    # local context solver non-simultaneous
    output_file="plots/CPU_times/error_vs_CPU_time_local_context_non_simult_experiment-1.pdf"
    input_dir="results/1/local_context_solver_non_simultaneous/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_context_non_simult_experiment-2_theta-${theta}.pdf"
    input_dir="results/2/${theta}/local_context_solver_non_simultaneous/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"

    output_file="plots/CPU_times/error_vs_CPU_time_local_context_non_simult_experiment-3_theta-${theta}.pdf"
    input_dir="results/3/${theta}/local_context_solver_non_simultaneous/"
    MPLBACKEND=Agg python plot_energy_norm_error_vs_time.py --path "${input_dir}" -o "${output_file}"
done