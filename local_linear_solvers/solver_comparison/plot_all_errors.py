import matplotlib.pyplot as plt
from pathlib import Path
from load_save_dumps import load_dump
import numpy as np


def main() -> None:
    base_result_paths = [
        Path('results/local_jacobi'),
        Path('results/local_block_jacobi'),
        # Path('results/local_gauss_seidel'),
        Path('results/local_context_solver_non_simultaneous'),
        Path('results/local_context_solver_simultaneous')]

    solver_names = [
        'jacobi',
        'block jacobi',
        # 'local gauss seidel',
        'context (non-simult.)',
        'context (simult.)']

    # ------------
    # read results
    # ------------
    energy_norm_errors_squared = []
    n_local_solves = []

    # looping over solvers
    for base_result_path in base_result_paths:
        # looping over iterates
        errs = []
        nsolvs = []
        for path_to_energy_norm_error in (
                base_result_path / Path('energy_norm_errors')).iterdir():
            nsolvs.append(int(path_to_energy_norm_error.stem))
            errs.append(load_dump(path_to_dump=path_to_energy_norm_error))
        energy_norm_errors_squared.append(errs)
        n_local_solves.append(nsolvs)

    energy_norm_errors_squared = np.array(energy_norm_errors_squared)
    n_local_solves = np.array(n_local_solves)

    # --------
    # plotting
    # --------
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams["figure.figsize"] = [6.4*1.5, 4.8*1.5]
    merged = []
    labels = []

    # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
    COLORS = [
        '#540D6E',
        '#ee4266',
        '#FFD23F',
        # '#0EAD69',
        '#3BCEAC']

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{local solves}}$')
    ax.set_ylabel(r'$\| u_n - u \|_a^2$')
    ax.grid(True)

    for errs, nsolvs, solver_name, color in zip(
            energy_norm_errors_squared, n_local_solves, solver_names, COLORS):
        sort_indices = nsolvs.argsort()

        errs = errs[sort_indices]
        nsolvs = nsolvs[sort_indices]

        line, = ax.loglog(
            nsolvs, errs, '--', linewidth=1.2, alpha=1, color=color)
        mark, = ax.loglog(
            nsolvs, errs, linestyle=None, marker='s',
            markersize=8, linewidth=0, alpha=0.6, color=color)
        merged.append((line, mark))
        labels.append(f'{solver_name}')

    ax.legend(merged, labels)
    fig.savefig('plots/solver_comparison.pdf', dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
