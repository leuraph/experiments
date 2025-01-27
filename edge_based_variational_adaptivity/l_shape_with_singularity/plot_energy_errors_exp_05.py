from load_save_dumps import load_dump
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re
from scipy.optimize import curve_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the result's `.pkl` files")
    parser.add_argument("-o", type=str, required=False,
                        default='energy_error_squared.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    path_to_exp = Path(args.path)
    output_path = Path(args.o)

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

    errs_arrays = []
    n_dofs_arrays = []
    thetas = []

    for theta_path in path_to_exp.iterdir():
        if not theta_path.is_dir():
            continue
        energy_norm_errors_squared_galerkin_with_orthogonality, n_dofs = \
            get_energy_norm_errors_squared_and_dofs(theta_path=theta_path)
        theta = get_theta_value_from_path(path=theta_path)

        if theta not in [0.1, 0.3, 0.5, 0.7, 0.9]:
            continue

        errs_arrays.append(
            energy_norm_errors_squared_galerkin_with_orthogonality)
        n_dofs_arrays.append(n_dofs)
        thetas.append(theta)

    # Loop through the pairs and plot each
    for i, (n_dofs, errs, theta) in enumerate(
            zip(n_dofs_arrays, errs_arrays, thetas)):
        # Generate a unique color for each pair
        color = plt.cm.tab10(i % 10)  # Cycle through a colormap (tab10)

        ax.loglog(
            n_dofs, errs,
            label=rf'$\theta = {theta}$',
            linestyle='--',  # Dotted line
            marker='s',     # Square markers
            color=color,    # Line and marker color
            markerfacecolor=color,  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=0.5,       # Transparency for markers
            markersize=4, linewidth=0.5
        )

    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs),
        np.log(errs))
    m_optimized = popt[0]

    ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
              'k--', linewidth=0.8)

    ax.legend(loc='best')

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def get_energy_norm_errors_squared_and_dofs(
        theta_path: Path) -> tuple[np.ndarray, np.ndarray]:

    energy_norm_errors_squared_galerkin_with_orthogonality = []
    n_dofs = []

    for n_dofs_dir in theta_path.iterdir():
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

    return energy_norm_errors_squared_galerkin_with_orthogonality, n_dofs


def get_theta_value_from_path(path: Path) -> float:
    # Regular expression to match 'theta_<number>'
    match = re.search(r"theta_([\d.]+)", str(path))

    if match:
        theta_value = float(match.group(1))  # Extract and convert to float
        return theta_value
    else:
        return None


if __name__ == '__main__':
    main()
