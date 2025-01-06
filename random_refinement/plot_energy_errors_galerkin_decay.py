from load_save_dumps import load_dump
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.optimize import curve_fit
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the folder including "
                        "all experiment folders")
    parser.add_argument("-o", type=str, required=False,
                        default='energy_error_squared.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    base_path = Path(args.path)
    output_path = Path(args.o)

    # Setup plot parameters
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16

    # preparing plot
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{DOF}}$')
    ax.set_ylabel(r'$\| u_h - u \|_a^2$')
    ax.grid(True)

    for percentage_dir in base_path.iterdir():
        if not percentage_dir.is_dir():
            continue

        energy_norm_errors_squared_galerkin_with_orthogonality = []
        n_dofs = []

        for n_dofs_dir in percentage_dir.iterdir():
            # exclude the initial solution
            if not n_dofs_dir.is_dir():
                continue

            n_dof = int(n_dofs_dir.name)
            path_to_energy_norm_error_squared_galerkin_with_orthogonality = \
                n_dofs_dir / Path(
                    'energy_norm_error_squared_galerkin_with_orthogonality.pkl')

            energy_norm_error_squared_galerkin_with_orthogonality = \
                load_dump(path_to_dump=path_to_energy_norm_error_squared_galerkin_with_orthogonality)

            n_dofs.append(n_dof)
            energy_norm_errors_squared_galerkin_with_orthogonality.append(
                energy_norm_error_squared_galerkin_with_orthogonality)

        energy_norm_errors_squared_galerkin_with_orthogonality = np.array(
            energy_norm_errors_squared_galerkin_with_orthogonality)
        n_dofs = np.array(n_dofs)

        sort_n_dof = n_dofs.argsort()

        energy_norm_errors_squared_galerkin_with_orthogonality = \
            energy_norm_errors_squared_galerkin_with_orthogonality[sort_n_dof]
        n_dofs = n_dofs[sort_n_dof]

        # Extract the numeric portion using a regex
        match = re.search(r'percentage_([\d\.]+)', percentage_dir.name)
        if match:
            percentage = float(match.group(1))
            print(f"Extracted percentage: {percentage}")
        else:
            print("Could not find a percentage in the folder name.")

        ax.loglog(
            n_dofs,
            energy_norm_errors_squared_galerkin_with_orthogonality,
            '--',
            marker='s',
            label=f'{percentage}',
            # markerfacecolor=(0, 0, 1, 0.5),
            markersize=4,
            linewidth=0.5)

        plot_model = percentage == 1.0
        if not plot_model:
            continue

        def model(x, m):
            return -x + m
        popt, pcov = curve_fit(
            model,
            np.log(n_dofs),
            np.log(energy_norm_errors_squared_galerkin_with_orthogonality))
        m_optimized = popt[0]

        ax.loglog(
            n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
            'k--', linewidth=0.8)

    ax.legend()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
