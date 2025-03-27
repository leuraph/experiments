from load_save_dumps import load_dump
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from p1afempy.solvers import get_stiffness_matrix
from scipy.sparse import csr_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the result's `.pkl` files")
    parser.add_argument("-o", type=str, required=True,
                        help="path to the outputted plot")
    args = parser.parse_args()

    results_path = Path(args.path)
    output_path = Path(args.o)

    energy_history = load_dump(
        path_to_dump=results_path / Path('energy_history.pkl'))
    iterate_history = load_dump(
        path_to_dump=results_path / Path('iterate_history.pkl'))

    coordinates = load_dump(results_path / Path('coordinates.pkl'))
    elements = load_dump(results_path / Path('elements.pkl'))

    stiffness_matrix = csr_matrix(get_stiffness_matrix(
        coordinates=coordinates, elements=elements))

    def get_energy_norm_squared(u: np.ndarray) -> float:
        return u.dot(stiffness_matrix.dot(u))

    # calculating relative changes in energy
    rel_energy = (
        (energy_history[1:] - energy_history[:-1]) /
        energy_history[:-1])

    # calculating relative changes of iterates in energy norm
    rel_norm = []
    for k in range(iterate_history.shape[0] - 1):
        old_u = iterate_history[k]
        new_u = iterate_history[k+1]

        numerator = get_energy_norm_squared(old_u - new_u)
        denonimator = get_energy_norm_squared(new_u)

        rel_norm.append(numerator / denonimator)

    # settinng plot params
    # --------------------

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{iterations}}$')
    ax.grid(True)

    ax.semilogy(
        np.abs(rel_energy),
        '--',
        alpha=0.5,
        label=r'relative change in energy')
    ax.semilogy(
        np.abs(rel_norm),
        '--',
        alpha=0.5,
        label=r'relative change in energy norm squared')

    ax.legend(loc='best')

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
