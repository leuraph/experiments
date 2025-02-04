from load_save_dumps import load_dump
from pathlib import Path
import argparse
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side
import matplotlib.pyplot as plt
from experiment_setup import f
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    base_result_path = Path(args.path)

    path_to_elements = base_result_path / Path('elements.pkl')
    path_to_coordinates = base_result_path / Path('coordinates.pkl')
    path_to_exact_solution = base_result_path / Path('exact_solution.pkl')
    elements = load_dump(path_to_dump=path_to_elements)
    coordinates = load_dump(path_to_dump=path_to_coordinates)
    exact_solution = load_dump(path_to_dump=path_to_exact_solution)

    stiffness_matrix = get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements)

    rhs = get_right_hand_side(
        coordinates=coordinates, elements=elements, f=f)

    exact_energy = 0.5 * exact_solution.dot(
        stiffness_matrix.dot(exact_solution)) \
        - exact_solution.dot(rhs)

    energies = []
    n_sweeps = []
    for n_sweep_dir in sorted(base_result_path.iterdir()):
        if n_sweep_dir.is_dir():
            path_to_iterate = n_sweep_dir / Path('solution.pkl')
            current_iterate = load_dump(path_to_dump=path_to_iterate)

            n_sweep = int(n_sweep_dir.name)
            n_sweeps.append(n_sweep)

            energy = 0.5 * current_iterate.dot(
                stiffness_matrix.dot(current_iterate)) \
                - current_iterate.dot(rhs)
            energies.append(energy)

    energies = np.array(energies)
    n_sweeps = np.array(n_sweeps)

    sort = np.argsort(n_sweeps)

    energies = energies[sort]
    n_sweeps = n_sweeps[sort]

    delta_energies = np.abs(energies - exact_energy)
    # --------
    # plotting
    # --------
    # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
    COLOR_RED = '#EE4266'
    COLOR_GREEN = '#0EAD69'

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams["figure.figsize"] = [6.4*1.5, 4.8*1.5]

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{sweeps}}$')
    ax.set_ylabel(r'$ | E(\widetilde{u}_n) - E(u_h) | $')
    ax.grid(True)

    ax.loglog(
        n_sweeps, delta_energies,
        '--', linewidth=1.2, alpha=0., color=COLOR_RED)
    ax.loglog(
        n_sweeps, delta_energies,
        linestyle=None, marker='s', markersize=8,
        linewidth=0, alpha=0.6, color=COLOR_RED)

    ax.hlines(exact_energy, xmin=0, xmax=20)

    plt.show()


if __name__ == '__main__':
    main()
