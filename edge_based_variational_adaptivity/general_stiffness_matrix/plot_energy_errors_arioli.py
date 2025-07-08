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
            energy_norm_errors_squared_last_iterate_to_exact, \
        n_dofs = get_energy_norm_errors_squared_and_dofs(
            results_path=results_path)
    n_iterations_on_each_mesh = get_n_iterations_on_each_mesh(
        results_path=results_path)
    delay_at_convergence_on_each_mesh = \
        get_delay_at_convergence_on_each_mesh(results_path=results_path)

    energy_norm_errors_galerkin_to_exact = np.sqrt(
        energy_norm_errors_squared_galerkin_to_exact)
    energy_norm_errors_last_iterate_to_galerkin = np.sqrt(
        energy_norm_errors_squared_last_iterate_to_galerkin)
    energy_norm_errors_last_iterate_to_exact = np.sqrt(
        energy_norm_errors_squared_last_iterate_to_exact)

    # settinng plot params
    # --------------------

    alpha_for_error_plots = 0.8

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots()
    ax.set_xlabel("$\mathrm{number}~\mathrm{of}~\mathrm{degrees}~\mathrm{of}~\mathrm{freedom}$")
    ax.set_ylabel("$\mathrm{energy}~\mathrm{norm}~\mathrm{error}$")
    ax.grid(True)

    # plotting ideal convergence order
    # --------------------------------
    def model(x, m):
        return -0.5*x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs),
        np.log(energy_norm_errors_last_iterate_to_exact))
    m_optimized = popt[0]
    ax.loglog(n_dofs, np.exp(model(np.log(n_dofs), m_optimized)),
              color='black', linestyle='--', linewidth=1.5)
    # --------------------------------

    # plotting |u_h - u_N^{n^\star}|_a
    # --------------------------------
    color = plt.cm.Set1(0)
    ax.loglog(
            n_dofs[1:],
            energy_norm_errors_last_iterate_to_galerkin[1:],
            linestyle='-',  # continuous line
            marker='s',     # Square markers
            color=color,    # Line and marker color
            markerfacecolor='none',  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=alpha_for_error_plots,       # Transparency for markers
            label='$\|u_h - u_N^{n^\star}\|_a$',
            markersize=8, linewidth=2.0)
    # -------------------------------------

    # plotting |u - u_h|_a
    # ----------------------
    # color = plt.cm.Set1(3)
    # ax.loglog(
    #         n_dofs, energy_norm_errors_galerkin_to_exact,
    #         linestyle='-',  # continuous line
    #         marker='v',     # Triangle down markers
    #         color=color,    # Line and marker color
    #         markerfacecolor='none',  # Marker fill color
    #         markeredgecolor=color,  # Marker outline color
    #         alpha=alpha_for_error_plots,       # Transparency for markers
    #         label='$\|u - u_h\|_a$',
    #         markersize=8, linewidth=1.0)
    # --------------------------------------

    # plotting |u - u_N^{n^\star}|_a
    # ----------------------
    color = plt.cm.Set1(1)
    ax.loglog(
            n_dofs, energy_norm_errors_last_iterate_to_exact,
            linestyle='-',  # continuous line
            marker='o',     # Circle markers
            color=color,    # Line and marker color
            markerfacecolor='none',  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=alpha_for_error_plots,       # Transparency for markers
            label='$\|u - u_N^{n^\star}\|_a$',
            markersize=8, linewidth=2.0)
    # --------------------------------------

    # plotting number of iterations and final delay on each mesh
    # ----------------------------------------------------------
    # Create a second y-axis for the second array 'b'
    color = plt.cm.Set1(2)
    ax_n_iterations = ax.twinx()
    ax_n_iterations.set_ylabel('$\mathrm{number}~\mathrm{of}~\mathrm{iterations}$')
    ax_n_iterations.plot(
        n_dofs, n_iterations_on_each_mesh,
        marker='D',  # Diamond marker
        linestyle=(0, (1, 5)),
        color=color,  # Fill color (RGB tuple)
        markerfacecolor='none',  # Marker fill color
        markeredgecolor=color,  # Marker outline color
        markersize=8, linewidth=2.0,
        label='$n_{\mathrm{iterations}}$'
    )
    # ax_n_iterations.scatter(
    #     n_dofs, delay_at_convergence_on_each_mesh,
    #     marker='D',  # Diamond marker
    #     c=[(0.4, 0.7, 0.3, 0.5)],  # Fill color (RGB tuple)
    #     edgecolors=[(0.4, 0.7, 0.3)],  # Outline color (same as fill)
    #     s=20,  # Marker size (optional, adjust as needed)
    #     label='$d$'
    # )
    ax_n_iterations.tick_params(axis='y')
    # ------------------------------------------

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
    energy_norm_errors_squared_last_iterate_to_exact: np.ndarray
        energy norm error squared of last iterate to the exact solution
        of the corresponding meshes
    n_dofs: np.ndarray
        number of degrees of freedom
    """

    energy_norm_errors_squared_galerkin_to_exact = []
    energy_norm_errors_squared_last_iterate_to_galerkin = []
    energy_norm_errors_squared_last_iterate_to_exact = []
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
        path_to_energy_norm_error_squared_last_iterate_to_exact = \
            n_dofs_dir / Path(
                'energy_norm_error_squared_last_iterate_to_exact.pkl')

        energy_norm_error_squared_galerkin_with_orthogonality = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_galerkin_with_orthogonality)
        energy_norm_error_squared_last_iterate_to_galerkin = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_last_iterate_to_galerkin)
        energy_norm_error_squared_last_iterate_to_exact = \
            load_dump(path_to_dump=path_to_energy_norm_error_squared_last_iterate_to_exact)

        n_dofs.append(n_dof)
        energy_norm_errors_squared_galerkin_to_exact.append(
            energy_norm_error_squared_galerkin_with_orthogonality)
        energy_norm_errors_squared_last_iterate_to_galerkin.append(
            energy_norm_error_squared_last_iterate_to_galerkin)
        energy_norm_errors_squared_last_iterate_to_exact.append(
            energy_norm_error_squared_last_iterate_to_exact)

    energy_norm_errors_squared_galerkin_to_exact = np.array(
        energy_norm_errors_squared_galerkin_to_exact)
    energy_norm_errors_squared_last_iterate_to_galerkin = np.array(
        energy_norm_errors_squared_last_iterate_to_galerkin)
    energy_norm_errors_squared_last_iterate_to_exact = np.array(
        energy_norm_errors_squared_last_iterate_to_exact
    )
    n_dofs = np.array(n_dofs)

    sort_n_dof = n_dofs.argsort()

    energy_norm_errors_squared_galerkin_to_exact = \
        energy_norm_errors_squared_galerkin_to_exact[sort_n_dof]
    energy_norm_errors_squared_last_iterate_to_galerkin = \
        energy_norm_errors_squared_last_iterate_to_galerkin[sort_n_dof]
    energy_norm_errors_squared_last_iterate_to_exact =\
        energy_norm_errors_squared_last_iterate_to_exact[sort_n_dof]
    n_dofs = n_dofs[sort_n_dof]

    return energy_norm_errors_squared_galerkin_to_exact, \
        energy_norm_errors_squared_last_iterate_to_galerkin, \
        energy_norm_errors_squared_last_iterate_to_exact, \
        n_dofs


def get_n_iterations_on_each_mesh(
        results_path: Path) -> np.ndarray:
    """
    reads and returns the number of iterations done on each mesh
    """

    n_iterations_on_each_mesh = []
    n_dofs = []

    for n_dofs_dir in results_path.iterdir():
        if not n_dofs_dir.is_dir():
            continue

        n_dof = int(n_dofs_dir.name)
        path_to_n_iterations_done = \
            n_dofs_dir / Path(
                'n_iterations_done.pkl')

        n_iterations_done = \
            load_dump(path_to_dump=path_to_n_iterations_done)

        n_dofs.append(n_dof)
        n_iterations_on_each_mesh.append(n_iterations_done)

    n_iterations_on_each_mesh = np.array(n_iterations_on_each_mesh)
    n_dofs = np.array(n_dofs)

    sort_n_dof = n_dofs.argsort()

    n_iterations_on_each_mesh = n_iterations_on_each_mesh[sort_n_dof]

    return n_iterations_on_each_mesh


def get_delay_at_convergence_on_each_mesh(
        results_path: Path) -> np.ndarray:
    """
    reads and returns the delay at convergence on each mesh
    """

    delay_at_convergence_on_each_mesh = []
    n_dofs = []

    for n_dofs_dir in results_path.iterdir():
        if not n_dofs_dir.is_dir():
            continue

        n_dof = int(n_dofs_dir.name)
        path_to_delay_at_convergence = \
            n_dofs_dir / Path(
                'delay_at_convergence.pkl')

        delay_at_convergence = \
            load_dump(path_to_dump=path_to_delay_at_convergence)

        n_dofs.append(n_dof)
        delay_at_convergence_on_each_mesh.append(delay_at_convergence)

    delay_at_convergence_on_each_mesh = \
        np.array(delay_at_convergence_on_each_mesh)
    n_dofs = np.array(n_dofs)

    sort_n_dof = n_dofs.argsort()

    delay_at_convergence_on_each_mesh = \
        delay_at_convergence_on_each_mesh[sort_n_dof]

    return delay_at_convergence_on_each_mesh


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
