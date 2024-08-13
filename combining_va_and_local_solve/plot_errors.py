import matplotlib.pyplot as plt
from pathlib import Path
from load_save_dumps import load_dump
import numpy as np
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, required=True,
                        help="path to plot to be generated")
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--fudge", type=float, required=True)
    args = parser.parse_args()

    theta = args.theta
    fudge = args.fudge
    path_to_plot = Path(args.o)

    base_result_path = Path(
        f'results/experiment_1/theta-{theta}_fudge-{fudge}')

    # ------------
    # read results
    # ------------
    energy_norm_errors_squared = []
    n_dofs = []

    # looping over iterates
    for path_to_results in base_result_path.iterdir():
        path_to_boundaries = path_to_results / Path('boundaries.pkl')
        path_to_coordinates = path_to_results / Path('coordinates.pkl')
        path_to_energy_norm_error = path_to_results / Path(
            'energy_norm_error.pkl')

        coordinates = load_dump(path_to_coordinates)
        boundaries = load_dump(path_to_boundaries)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))

        energy_norm_errors_squared.append(load_dump(path_to_energy_norm_error))
        n_dofs.append(len(indices_of_free_nodes))

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
    merged = []
    # labels = []

    # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
    COLOR = '#ee4266'

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{dof}}$')
    ax.set_ylabel(r'$\| u_n - u_h \|_a^2$')
    ax.grid(True)

    sort = np.flip(np.argsort(energy_norm_errors_squared))
    n_dofs = n_dofs[sort]
    energy_norm_errors_squared = energy_norm_errors_squared[sort]

    e_star = energy_norm_errors_squared[1]
    n_star = n_dofs[1]
    ideal_convergence = lambda n: e_star * n_star / n

    ax.loglog(
        np.unique(n_dofs), ideal_convergence(np.unique(n_dofs)),
        linestyle='--',
        linewidth=1, alpha=0.6, color='black')

    line, = ax.loglog(
        n_dofs, energy_norm_errors_squared,
        '--', linewidth=1.2, alpha=0., color=COLOR)
    mark, = ax.loglog(
        n_dofs, energy_norm_errors_squared,
        linestyle=None, marker='s', markersize=8,
        linewidth=0, alpha=0.6, color=COLOR)
    merged.append((line, mark))

    # ax.legend(merged, labels)
    fig.savefig(
        path_to_plot,
        dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
