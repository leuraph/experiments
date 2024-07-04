import matplotlib.pyplot as plt
from pathlib import Path
from load_save_dumps import load_dump
import numpy as np
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True,
        help=(
            "path to folder holding 'elapsed_times' "
            "and 'energy_norm_errors' directories"))
    parser.add_argument("-o", type=str, required=True,
                        help="path to plot to be generated")
    args = parser.parse_args()
    path_to_dirs = Path(args.path)
    path_to_plot = Path(args.o)

    # ------------
    # read results
    # ------------
    energy_norm_errors_squared: list[float] = []
    elapsed_times_s: list[float] = []

    # loading errors
    # --------------
    n_solves: list[int] = []  # used for sorting only
    path_to_errs = path_to_dirs / Path('energy_norm_errors')
    for path_to_energy_norm_error in path_to_errs.iterdir():
        energy_norm_errors_squared.append(
            load_dump(path_to_dump=path_to_energy_norm_error))
        n_solves.append(int(path_to_energy_norm_error.stem))

    # array conversion and sorting based on n_local_solves
    energy_norm_errors_squared = np.array(energy_norm_errors_squared)
    n_solves = np.array(n_solves)
    energy_norm_errors_squared = energy_norm_errors_squared[
        np.argsort(n_solves)]

    # loading elapsed times (s)
    # -------------------------
    n_solves: list[int] = []  # used for sorting only
    path_to_elapsed_times = path_to_dirs / Path('elapsed_times')
    for path_to_elapsed_time_s in path_to_elapsed_times.iterdir():
        elapsed_times_s.append(load_dump(path_to_elapsed_time_s))
        n_solves.append(int(path_to_elapsed_time_s.stem))

    # array conversion and sorting based on n_local_solves
    elapsed_times_s = np.array(elapsed_times_s)
    n_solves = np.array(n_solves)
    elapsed_times_s = elapsed_times_s[np.argsort(n_solves)]

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
    # merged = []
    # labels = []

    # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
    COLORS = [
        '#540D6E',
        '#ee4266',
        '#FFD23F',
        # '#0EAD69',
        '#3BCEAC']

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$\| u_n - u_h \|_a^2$')
    ax.set_ylabel(r'$\Delta t$ / s')
    ax.grid(True)

    color = '#540D6E'
    line, = ax.loglog(
        energy_norm_errors_squared, elapsed_times_s,
        '--', linewidth=1.2, alpha=1, color=color)
    mark, = ax.loglog(
        energy_norm_errors_squared, elapsed_times_s,
        linestyle=None, marker='s',
        markersize=8, linewidth=0, alpha=0.6, color=color)
    # merged.append((line, mark))
    # labels.append(f'{solver_name}')

    # ax.legend(merged, labels)
    fig.savefig(
        path_to_plot,
        dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
