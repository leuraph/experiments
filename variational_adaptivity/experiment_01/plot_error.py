import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from p1afempy import solvers
from scipy.optimize import curve_fit


def main() -> None:

    THETA = 0.5

    base_path = Path(f'results/theta_{THETA}/')

    exact_energy_squared = 0.01305598516695022655

    energies_squared = []
    n_elements = []
    n_coordinates = []
    n_dofs = []

    for n_dir in base_path.iterdir():
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

    errs_squared = exact_energy_squared - energies_squared

    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(model, np.log(n_dofs), np.log(errs_squared))
    m_optimized = popt[0]

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    fig, ax = plt.subplots()
    ax.set_title(fr'Variational Adaptivity (VA) $\theta=$ {THETA}')
    ax.set_xlabel(r'$n_{\text{DOF}}$')
    ax.set_ylabel(r'$\| u_h - u \|_a^2$')
    ax.grid(True)
    ax.loglog(n_dofs, errs_squared, 'b--', marker='s', markerfacecolor=(0, 0, 1, 0.5), markersize=4, linewidth=0.5)
    ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)), 'k--', linewidth=0.8) #, label=r'$\propto -\log n_{\text{elements}} $')
    ax.legend()

    fig.savefig(f'energy_error_theta_{THETA}.png', dpi=300)


if __name__ == '__main__':
    main()
