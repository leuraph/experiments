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

    results_path = Path(args.path)
    output_path = Path(args.o)

    energy_norm_errors_squared_galerkin_to_exact, \
        energy_norm_errors_squared_last_iterate_to_galerkin, \
        n_dofs = get_energy_norm_errors_squared_and_dofs(
            results_path=results_path)

    # settinng plot params
    # --------------------

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{DOF}}$')
    # ax.set_ylabel(r'$\| u_h - u_N^{n^\star} \|_a^2$')
    ax.grid(True)

    # plotting |u_h - u_N^{n^\star}|_a^2
    # ----------------------------------
    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs[1:]),
        np.log(energy_norm_errors_squared_last_iterate_to_galerkin[1:]))
    m_optimized = popt[0]

    color = plt.cm.tab10(1)
    ax.loglog(
            n_dofs[1:],
            energy_norm_errors_squared_last_iterate_to_galerkin[1:],
            linestyle='--',  # Dotted line
            marker='s',     # Square markers
            color=color,    # Line and marker color
            markerfacecolor=color,  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=0.5,       # Transparency for markers
            label=r'$\|u_h - u_N^{n^\star}\|^2_a$',
            markersize=5, linewidth=1.0)
    ax.loglog(n_dofs[1:], np.exp(model(np.log(n_dofs[1:]), m_optimized)),
              'k--', linewidth=0.8)
    # -------------------------------------

    # plotting |u - u_h|_a^2
    # ----------------------
    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs),
        np.log(energy_norm_errors_squared_galerkin_to_exact))
    m_optimized = popt[0]

    color = plt.cm.tab10(2)
    ax.loglog(
            n_dofs, energy_norm_errors_squared_galerkin_to_exact,
            linestyle='--',  # Dotted line
            marker='s',     # Square markers
            color=color,    # Line and marker color
            markerfacecolor=color,  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=0.5,       # Transparency for markers
            label=r'$\|u - u_h\|^2_a$',
            markersize=5, linewidth=1.0)
    ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
              'k--', linewidth=0.8)

    ax.legend(loc='best')

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def get_energy_norm_errors_squared_and_dofs(
        results_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    reads and returns the energy norm errors squared of
    the Galerkin solution and the last iterate.

    note
    ----
    as the exact solution u(x, y) is unavailbale in this experiment,
    we need to calculate both |u - u_h| and |u_n - u_h| separately.

    returns
    -------
    energy_norm_errors_squared_galerkin_to_exact: np.ndarray
        energy norm errors squared of Galerkin solutions to the exact solution
        using Galerkin Orthogonality
    energy_norm_errors_squared_last_iterate_to_galerkin: np.ndarray
        energy norm error squared of last iterate to the Galerkin solution
        of the corresponding meshes
    n_dofs: np.ndarray
        number of degrees of freedom
    """

    energy_norm_errors_squared_galerkin_to_exact = []
    energy_norm_errors_squared_last_iterate_to_galerkin = []
    n_dofs = []

    for n_dofs_dir in results_path.iterdir():
        if not n_dofs_dir.is_dir():
            continue

        n_dof = int(n_dofs_dir.name)
        path_to_energy_norm_error_squared_galerkin_with_orthogonality = \
            n_dofs_dir / Path(
                'energy_norm_error_squared_galerkin_with_orthogonality.pkl')
        path_to_energy_norm_error_squared_last_iterate_to_galerkin = \
            n_dofs_dir / Path(
                'energy_norm_error_squared_last_iterate_to_galerkin.pkl')

        energy_norm_error_squared_galerkin_with_orthogonality = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_galerkin_with_orthogonality)
        energy_norm_error_squared_last_iterate_to_galerkin = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_last_iterate_to_galerkin)

        n_dofs.append(n_dof)
        energy_norm_errors_squared_galerkin_to_exact.append(
            energy_norm_error_squared_galerkin_with_orthogonality)
        energy_norm_errors_squared_last_iterate_to_galerkin.append(
            energy_norm_error_squared_last_iterate_to_galerkin)

    energy_norm_errors_squared_galerkin_to_exact = np.array(
        energy_norm_errors_squared_galerkin_to_exact)
    energy_norm_errors_squared_last_iterate_to_galerkin = np.array(
        energy_norm_errors_squared_last_iterate_to_galerkin)
    n_dofs = np.array(n_dofs)

    sort_n_dof = n_dofs.argsort()

    energy_norm_errors_squared_galerkin_to_exact = \
        energy_norm_errors_squared_galerkin_to_exact[sort_n_dof]
    energy_norm_errors_squared_last_iterate_to_galerkin = \
        energy_norm_errors_squared_last_iterate_to_galerkin[sort_n_dof]
    n_dofs = n_dofs[sort_n_dof]

    return energy_norm_errors_squared_galerkin_to_exact, \
        energy_norm_errors_squared_last_iterate_to_galerkin, \
        n_dofs


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
