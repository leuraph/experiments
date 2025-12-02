import argparse
from pathlib import Path
from tqdm import tqdm
from load_save_dumps import load_dump
from p1afempy.io_helpers import read_elements, read_coordinates
import numpy as np
from problems import get_problem
import matplotlib.pyplot as plt
from compute_energies import compute_energy
from scipy.optimize import curve_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-energy", type=str, required=True)
    parser.add_argument("--problem", type=int, required=True)
    parser.add_argument("--default-path", type=str, required=True)
    parser.add_argument("--arioli-path", type=str, required=True)
    parser.add_argument("--tail-off-path", type=str, required=True)
    args = parser.parse_args()

    problem = get_problem(number=args.problem)

    path_to_reference_energy = Path(args.reference_energy)
    path_to_default_results = Path(args.default_path)
    path_to_arioli_results = Path(args.arioli_path)
    path_to_tail_off_results = Path(args.tail_off_path)

    file_name = Path(
        f'problem-{args.problem}.pdf')
    
    output_path = Path('plots') / file_name

    reference_energy = load_dump(
        path_to_dump=path_to_reference_energy)

    # STOPPING CRITERIA
    # -----------------
    energy_differences_default, n_dofs_default, n_iterations_default = get_energy_diffs(
        path_to_results=path_to_default_results,
        reference_energy=reference_energy)
    energy_differences_arioli, n_dofs_arioli, n_iterations_arioli = get_energy_diffs(
        path_to_results=path_to_arioli_results,
        reference_energy=reference_energy)
    energy_differences_tail_off, n_dofs_tail_off, n_iterations_tail_off = get_energy_diffs(
        path_to_results=path_to_tail_off_results,
        reference_energy=reference_energy)
    # -----------------

    # settinng plot params
    # --------------------

    # in the paper, we show two figures side by side
    # including a graphic of (possibly) smaller witdth than 1.0\textwidth
    # inside a minipage of (possibly) smaller width than 0.5\textwidth,
    # hence, the scaling factor:
    graphic_scaling = 0.7
    minipage_scaling = 1.0
    paper_scaling = 1/(graphic_scaling * minipage_scaling)
    alpha_for_error_plots = 0.8
    font_size = 14 * paper_scaling

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$n_{\text{DOF}}$')
    ax.set_ylabel(r'$E(\tilde u) - E(u^\star_N)$')
    ax.grid(True)

    color = plt.cm.Set1(0)
    ax.loglog(
            n_dofs_default[1:],
            energy_differences_default[1:],
            linestyle='-',  # continuous line
            marker='s',     # Square markers
            color=color,    # Line and marker color
            markerfacecolor='none',  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=alpha_for_error_plots,       # Transparency for markers
            markersize=8*paper_scaling,
            linewidth=2.0*paper_scaling,
            label='default')

    color = plt.cm.Set1(1)
    ax.loglog(
            n_dofs_arioli[1:],
            energy_differences_arioli[1:],
            linestyle='-',  # continuous line
            marker='o',     # Circle markers
            color=color,    # Line and marker color
            markerfacecolor='none',  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=alpha_for_error_plots,       # Transparency for markers
            markersize=8*paper_scaling,
            linewidth=2.0*paper_scaling,
            label='relative decay')

    color = plt.cm.Set1(2)
    ax.loglog(
            n_dofs_tail_off[1:],
            energy_differences_tail_off[1:],
            linestyle='-',  # continuous line
            marker='v',     # Triangle markers
            color=color,    # Line and marker color
            markerfacecolor='none',  # Marker fill color
            markeredgecolor=color,  # Marker outline color
            alpha=alpha_for_error_plots,       # Transparency for markers
            markersize=8*paper_scaling,
            linewidth=2.0*paper_scaling,
            label='tail-off')

    # plotting ideal convergence order
    # --------------------------------
    n_points_for_fit = 2
    def model(x, m):
        return -x + m
    popt, pcov = curve_fit(
        model,
        np.log(n_dofs_tail_off[-n_points_for_fit:]),
        np.log(energy_differences_tail_off[-n_points_for_fit:]))
    m_optimized = popt[0]
    ax.loglog(n_dofs_default[1:], np.exp(model(np.log(n_dofs_default[1:]), m_optimized)),
              color='black', linestyle='--',
              linewidth=1.5)
    # --------------------------------

    # # plotting number of iterations on each mesh
    # # ------------------------------------------
    # # Create a second y-axis for the second array 'b'
    # ax_n_iterations = ax.twinx()
    # ax_n_iterations.set_ylabel('$\mathrm{number}~\mathrm{of}~\mathrm{iterations}$')

    # color = plt.cm.Set1(0)
    # ax_n_iterations.plot(
    #     n_dofs_default, n_iterations_default,
    #     marker='s',  # Square marker
    #     linestyle=(0, (1, 5)),
    #     color=color,  # Fill color (RGB tuple)
    #     markerfacecolor='none',  # Marker fill color
    #     markeredgecolor=color,  # Marker outline color
    #     markersize=8*paper_scaling,
    #     linewidth=2.0*paper_scaling,
    #     label='$n_{\mathrm{iterations}}$'
    # )

    # color = plt.cm.Set1(1)
    # ax_n_iterations.plot(
    #     n_dofs_arioli, n_iterations_arioli,
    #     marker='o',  # Circle marker
    #     linestyle=(0, (1, 5)),
    #     color=color,  # Fill color (RGB tuple)
    #     markerfacecolor='none',  # Marker fill color
    #     markeredgecolor=color,  # Marker outline color
    #     markersize=8*paper_scaling,
    #     linewidth=2.0*paper_scaling,
    #     label='$n_{\mathrm{iterations}}$'
    # )

    # color = plt.cm.Set1(2)
    # ax_n_iterations.plot(
    #     n_dofs_tail_off, n_iterations_tail_off,
    #     marker='v',  # Triangle marker
    #     linestyle=(0, (1, 5)),
    #     color=color,  # Fill color (RGB tuple)
    #     markerfacecolor='none',  # Marker fill color
    #     markeredgecolor=color,  # Marker outline color
    #     markersize=8*paper_scaling,
    #     linewidth=2.0*paper_scaling,
    #     label='$n_{\mathrm{iterations}}$'
    # )
    # # ------------------------------------------

    ax.legend(loc='best')

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def get_energy_diffs(
        path_to_results: Path,
        reference_energy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    returns the energy differences and corresponding
    number of degrees of freedom
    """

    energies = []
    n_dofs = []
    n_iterations = []

    for path_to_n_dofs in tqdm(list(path_to_results.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue 
        n_dof = int(path_to_n_dofs.name)
        n_dofs.append(n_dof)

        current_energy = load_dump(
            path_to_dump=path_to_n_dofs / 'energy.pkl')
        current_n_iterations = load_dump(
                path_to_dump=path_to_n_dofs / 'n_iterations.pkl')
        
        energies.append(current_energy)
        n_iterations.append(current_n_iterations)

    # converting lists to numpy arrays
    energies = np.array(energies)
    n_dofs = np.array(n_dofs)
    n_iterations = np.array(n_iterations)
    
    # sorting corresponding to number of degrees of freedom
    sort_n_dof = n_dofs.argsort()
    n_dofs = n_dofs[sort_n_dof]
    energies = energies[sort_n_dof]
    n_iterations = n_iterations[sort_n_dof]

    energy_differences = energies - reference_energy
    if np.any(energy_differences < 0.0):
        raise RuntimeError(
            "None of the energy differences should be negative!" \
            "This probably means that the reference mesh was too " \
            "coarse or the reference solution too rough"
        )

    return energy_differences, n_dofs, n_iterations


if __name__ == '__main__':
    main()
