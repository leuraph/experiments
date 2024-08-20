import matplotlib.pyplot as plt
from pathlib import Path
from load_save_dumps import load_dump
import numpy as np
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, required=True,
                        help="path to plot to be generated")
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    path_to_plot = Path(args.o)
    base_result_path = Path(args.path)

    # ------------
    # read results
    # ------------
    energy_norm_errors_squared = []
    energy_norm_error_squared_exact_solutions = []
    n_dofs = []

    for n_dofs_dir in base_result_path.iterdir():

        path_to_boundaries = n_dofs_dir / Path('boundaries.pkl')
        path_to_coordinates = n_dofs_dir / Path('coordinates.pkl')
        path_to_energy_norm_error_squared_exact = n_dofs_dir / Path(
            'energy_norm_error_squared_exact.pkl')

        coordinates = load_dump(path_to_coordinates)
        boundaries = load_dump(path_to_boundaries)
        energy_norm_error_squared_exact = load_dump(
            path_to_energy_norm_error_squared_exact)

        energy_norm_error_squared_exact_solutions.append(
            energy_norm_error_squared_exact)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))

        n_dofs.append(len(indices_of_free_nodes))

        errs = []
        for n_sweeps_dir in n_dofs_dir.iterdir():
            if n_sweeps_dir.is_dir():
                energy_norm_error_squared = load_dump(
                    path_to_dump=n_sweeps_dir/"energy_norm_error_squared.pkl")
                errs.append(energy_norm_error_squared)
        energy_norm_errors_squared.append(errs)

    energy_norm_errors_squared = np.array(energy_norm_errors_squared)
    energy_norm_error_squared_exact_solutions = np.array(
        energy_norm_error_squared_exact_solutions)
    n_dofs = np.array(n_dofs)

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
    # labels = []

    # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
    COLOR_RED = '#EE4266'
    COLOR_GREEN = '#0EAD69'

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{dof}}$')
    ax.set_ylabel(r'$\| \widetilde{u} - u \|_a^2$')
    ax.grid(True)

    sort = np.argsort(n_dofs)
    n_dofs = n_dofs[sort]
    energy_norm_error_squared_exact_solutions = \
        energy_norm_error_squared_exact_solutions[sort]
    energy_norm_errors_squared = energy_norm_errors_squared[sort]

    e_star = energy_norm_error_squared_exact_solutions[1]
    n_star = n_dofs[1]

    def ideal_convergence(n):
        return e_star * n_star / n

    ax.loglog(
        np.unique(n_dofs), ideal_convergence(np.unique(n_dofs)),
        linestyle='--',
        linewidth=1, alpha=0.6, color='black')

    for k, n_dof in enumerate(n_dofs):
        mark, = ax.loglog(
            n_dof*np.ones_like(energy_norm_errors_squared[k]),
            energy_norm_errors_squared[k],
            linestyle=None, marker='_', markersize=8,
            linewidth=0, alpha=1.0, color=COLOR_GREEN)

    line, = ax.loglog(
        n_dofs, energy_norm_error_squared_exact_solutions,
        '--', linewidth=1.2, alpha=0., color=COLOR_RED)
    mark, = ax.loglog(
        n_dofs, energy_norm_error_squared_exact_solutions,
        linestyle=None, marker='_', markersize=8,
        linewidth=0, alpha=0.6, color=COLOR_RED)
    merged.append((line, mark))

    # ax.legend(merged, labels)
    fig.savefig(
        path_to_plot,
        dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
