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

    max_n_sweeps = 1000
    min_n_sweeps = 50

    # ------------
    # read results
    # ------------
    n_dofs = []
    n_sweeps = []

    for n_dofs_dir in base_result_path.iterdir():

        path_to_boundaries = n_dofs_dir / Path('boundaries.pkl')
        path_to_coordinates = n_dofs_dir / Path('coordinates.pkl')

        coordinates = load_dump(path_to_coordinates)
        boundaries = load_dump(path_to_boundaries)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))

        n_dofs.append(len(indices_of_free_nodes))

        n_sweeps_on_this_mesh = 0
        for n_sweeps_dir in n_dofs_dir.iterdir():
            if n_sweeps_dir.is_dir():
                n_sweep_of_this_dir = int(n_sweeps_dir.name)
                if n_sweeps_on_this_mesh < n_sweep_of_this_dir:
                    n_sweeps_on_this_mesh = n_sweep_of_this_dir
        n_sweeps.append(n_sweeps_on_this_mesh)

    n_dofs = np.array(n_dofs)
    n_sweeps = np.array(n_sweeps)

    sort = np.argsort(n_dofs)

    n_dofs = n_dofs[sort]
    n_sweeps = n_sweeps[sort]

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
    ax.set_ylabel(r'$n_{\text{sweeps}}$')
    ax.grid(True)

    line, = ax.loglog(
        n_dofs, n_sweeps,
        '--', linewidth=1.2, alpha=0., color=COLOR_RED)
    mark, = ax.loglog(
        n_dofs, n_sweeps,
        linestyle=None, marker='s', markersize=8,
        linewidth=0, alpha=0.6, color=COLOR_RED)
    merged.append((line, mark))

    ax.hlines(
        y=max_n_sweeps, xmin=n_dofs[0], xmax=n_dofs[-1], colors='red',
        label=r'$n_{\text{sweeps, max}}$')
    ax.hlines(
        y=min_n_sweeps, xmin=n_dofs[0], xmax=n_dofs[-1], colors='green',
        label=r'$n_{\text{sweeps, min}}$')

    ax.set_xlim(
        left=1e1,
        right=1e6)
    ax.set_ylim(
        bottom=min_n_sweeps * 0.9,
        top=max_n_sweeps * 1.1)

    ax.legend()
    fig.savefig(
        path_to_plot,
        dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
