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

    energy_norm_errors_squared = []
    energy_norm_errors_squared_exact = []
    n_dofs = []

    for n_dofs_dir in base_result_path.iterdir():
        path_to_boundaries = n_dofs_dir / Path('boundaries.pkl')
        path_to_coordinates = n_dofs_dir / Path('coordinates.pkl')
        path_to_energy_norm_error_squared = n_dofs_dir / Path(
            'energy_norm_error_squared_last_iterate.pkl')
        path_to_energy_norm_error_squared_exact = n_dofs_dir / Path(
            'energy_norm_error_squared_galerkin_without_orthogonality.pkl')

        coordinates = load_dump(path_to_coordinates)
        boundaries = load_dump(path_to_boundaries)
        energy_norm_error_squared_exact = load_dump(
            path_to_energy_norm_error_squared_exact)
        energy_norm_error_squared = load_dump(
            path_to_energy_norm_error_squared)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))

        energy_norm_errors_squared.append(energy_norm_error_squared)
        energy_norm_errors_squared_exact.append(
            energy_norm_error_squared_exact)
        n_dofs.append(len(indices_of_free_nodes))

    energy_norm_errors_squared_exact = np.array(
        energy_norm_errors_squared_exact)
    energy_norm_errors_squared = np.array(energy_norm_errors_squared)
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

    # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
    COLOR_RED = '#EE4266'
    COLOR_GREEN = '#0EAD69'

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{dof}}$')
    ax.set_ylabel(r'$\| \widetilde{u} - u \|_a^2$')
    ax.grid(True)

    sort = np.argsort(n_dofs)

    n_dofs = n_dofs[sort]

    energy_norm_errors_squared_exact = \
        energy_norm_errors_squared_exact[sort]

    energy_norm_errors_squared = energy_norm_errors_squared[sort]

    e_star = energy_norm_errors_squared_exact[1]
    n_star = n_dofs[1]

    def ideal_convergence(n):
        return e_star * n_star / n

    ax.loglog(
        n_dofs, ideal_convergence(n_dofs),
        linestyle='--',
        linewidth=1, alpha=0.6, color='black')

    ax.loglog(
        n_dofs,
        energy_norm_errors_squared_exact,
        linestyle=None, marker='_', markersize=8,
        linewidth=0, alpha=1.0, color=COLOR_RED,
        label='Galerkin solutions')

    ax.loglog(
        n_dofs, energy_norm_errors_squared,
        linestyle=None, marker='_', markersize=8,
        linewidth=0, alpha=1.0, color=COLOR_GREEN,
        label='Last iterate')
    
    ax.legend(loc='best')

    fig.savefig(
        path_to_plot,
        dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
