from load_save_dumps import load_dump
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.optimize import curve_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the result's `.pkl` files")
    parser.add_argument("-o", type=str, required=False,
                        default='energy_error_squared.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    base_path = Path(args.path)
    output_path = Path(args.o)

    energy_norm_errors_squared_galerkin_with_orthogonality = []
    energy_norm_errors_squared_galerkin_without_orthogonality = []
    n_dofs = []

    for n_dofs_dir in base_path.iterdir():
        # exclude the initial solution
        if not n_dofs_dir.is_dir():
            continue

        n_dof = int(n_dofs_dir.name)
        path_to_energy_norm_error_squared_galerkin_with_orthogonality = \
            n_dofs_dir / Path(
                'energy_norm_error_squared_galerkin_with_orthogonality.pkl')
        path_to_energy_norm_error_squared_galerkin_without_orthogonality = \
            n_dofs_dir / Path(
                'energy_norm_error_squared_galerkin_without_orthogonality.pkl')    

        energy_norm_error_squared_galerkin_with_orthogonality = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_galerkin_with_orthogonality)
        energy_norm_error_squared_galerkin_without_orthogonality = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_galerkin_without_orthogonality)

        n_dofs.append(n_dof)
        energy_norm_errors_squared_galerkin_with_orthogonality.append(
            energy_norm_error_squared_galerkin_with_orthogonality)
        energy_norm_errors_squared_galerkin_without_orthogonality.append(
            energy_norm_error_squared_galerkin_without_orthogonality)

    energy_norm_errors_squared_galerkin_with_orthogonality = np.array(
        energy_norm_errors_squared_galerkin_with_orthogonality)
    energy_norm_errors_squared_galerkin_without_orthogonality = np.array(
        energy_norm_errors_squared_galerkin_without_orthogonality)
    n_dofs = np.array(n_dofs)

    sort_n_dof = n_dofs.argsort()

    energy_norm_errors_squared_galerkin_with_orthogonality = \
        energy_norm_errors_squared_galerkin_with_orthogonality[sort_n_dof]
    energy_norm_errors_squared_galerkin_without_orthogonality = \
        energy_norm_errors_squared_galerkin_without_orthogonality[sort_n_dof]
    n_dofs = n_dofs[sort_n_dof]

    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs),
        np.log(energy_norm_errors_squared_galerkin_with_orthogonality))
    m_optimized = popt[0]

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
    ax.loglog(
        n_dofs, energy_norm_errors_squared_galerkin_with_orthogonality,
        'b--', marker='s', label='with orthogonality',
        markerfacecolor=(0, 0, 1, 0.5), markersize=4, linewidth=0.5)
    ax.loglog(
        n_dofs, energy_norm_errors_squared_galerkin_without_orthogonality,
        'r--', marker='s', label='without orthogonality',
        markerfacecolor=(0, 0, 1, 0.5), markersize=4, linewidth=0.5)
    ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
              'k--', linewidth=0.8)
    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
