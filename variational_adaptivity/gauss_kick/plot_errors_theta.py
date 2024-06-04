import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from p1afempy import solvers
from scipy.optimize import curve_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, required=True,
                        help="path to the experiment's results directory "
                        "holding the directories corresponding to "
                        "different Dörfler parameters (theta's)")
    parser.add_argument("--energy-path", type=str, required=True,
                        help="path to the file holding the numerical value of "
                        "the solution's energy norm squared")
    parser.add_argument("-o", type=str, required=False,
                        default='energy_errors_squared.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    with open(args.energy_path) as f:
        energy_squared_exact = float(f.readline())

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{DOF}}$')
    ax.set_ylabel(r'$\| u_h - u \|_a^2$')
    ax.grid(True)

    THETAS = [0.6, 0.5, 0.4]
    COLORS = ['#2ec4b6', '#e71d36', '#ff9f1c']
    merged = []
    labels = []
    for k, THETA in enumerate(THETAS):

        base_path = Path(args.results_path) / Path(f'theta_{THETA}/')

        energies_squared = []
        n_elements = []
        n_coordinates = []
        n_dofs = []

        for n_dir in base_path.iterdir():
            # exclude the initial solution
            if not n_dir.is_dir() or n_dir.name == '0':
                continue
            path_to_coordinates = \
                n_dir / Path('coordinates.pkl')
            path_to_elements = \
                n_dir / Path('elements.pkl')
            path_to_solution = \
                n_dir / Path('solution.pkl')
            path_to_boundaries = \
                n_dir / Path('boundaries.pkl')

            with open(path_to_coordinates, mode='rb') as file:
                coordinates = pickle.load(file=file)
            with open(path_to_elements, mode='rb') as file:
                elements = pickle.load(file=file)
            with open(path_to_solution, mode='rb') as file:
                solution = pickle.load(file=file)
            with open(path_to_boundaries, mode='rb') as file:
                boundaries = pickle.load(file=file)[0]

            stiffness_matrix = solvers.get_stiffness_matrix(
                coordinates=coordinates, elements=elements)

            energy_squared = solution.dot(stiffness_matrix.dot(solution))

            energies_squared.append(energy_squared)
            n_elements.append(elements.shape[0])
            n_coordinates.append(coordinates.shape[0])
            n_dofs.append(coordinates.shape[0] - (boundaries.shape[0] + 1))

        energies_squared = np.array(energies_squared)
        n_elements = np.array(n_elements)
        n_coordinates = np.array(n_coordinates)
        n_dofs = np.array(n_dofs)

        sort_indices = n_dofs.argsort()

        energies_squared = energies_squared[sort_indices]
        n_dofs = n_dofs[sort_indices]

        errs_squared = energy_squared_exact - energies_squared

        line, = ax.loglog(n_dofs, errs_squared, '--', linewidth=1.2,
                          color=COLORS[k], alpha=1)
        mark, = ax.loglog(n_dofs, errs_squared, linestyle=None, marker='s',
                          markersize=8, linewidth=0, alpha=0.3,
                          color=COLORS[k])
        merged.append((line, mark))
        labels.append(fr'$\theta = ${THETA}')

        if THETA == max(THETAS):
            # fitting y = -x + q to the highest value of THETA
            def model(x, m):
                return -x + m
            popt, pcov = curve_fit(model, np.log(n_dofs), np.log(errs_squared))
            m_optimized = popt[0]
            ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
                      'k--', linewidth=1)

    ax.legend(merged, labels)
    fig.savefig(args.o, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
