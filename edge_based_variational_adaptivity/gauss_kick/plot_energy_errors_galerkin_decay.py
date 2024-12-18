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
    parser.add_argument("--energy-path", type=str, required=True,
                        help="path to the file holding the numerical value of "
                        "the solution's energy norm squared")
    parser.add_argument("-o", type=str, required=False,
                        default='energy_error_squared.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    with open(args.energy_path) as f:
        energy_norm_squared_exact = float(f.readline())

    base_path = Path(args.path)
    output_path = Path(args.o)

    energy_norms_squared_galerkin = []
    n_dofs = []

    for n_dofs_dir in base_path.iterdir():
        # exclude the initial solution
        if not n_dofs_dir.is_dir():
            continue

        n_dof = int(n_dofs_dir.name)
        path_to_energy_norm_squared_galerkin = \
            n_dofs_dir / Path('energy_norm_squared.pkl')

        energy_norm_squared_galerkin = load_dump(
            path_to_dump=path_to_energy_norm_squared_galerkin)

        n_dofs.append(n_dof)
        energy_norms_squared_galerkin.append(energy_norm_squared_galerkin)

    energy_norms_squared_galerkin = np.array(energy_norms_squared_galerkin)
    n_dofs = np.array(n_dofs)

    sort_n_dof = n_dofs.argsort()

    energy_norms_squared_galerkin = energy_norms_squared_galerkin[sort_n_dof]
    n_dofs = n_dofs[sort_n_dof]

    energy_errors_squared =\
        energy_norm_squared_exact - energy_norms_squared_galerkin

    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs),
        np.log(energy_errors_squared))
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
    ax.loglog(n_dofs, energy_errors_squared, 'b--', marker='s',
              markerfacecolor=(0, 0, 1, 0.5), markersize=4, linewidth=0.5)
    ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
              'k--', linewidth=0.8)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
