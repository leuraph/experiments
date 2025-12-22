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
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    path_to_reference_energy = Path(args.reference_energy)
    path_to_results = Path(args.path)

    file_name = Path(
        f'{path_to_results.name}.pdf')
    
    output_path = Path('plots') / file_name

    reference_energy = load_dump(
        path_to_dump=path_to_reference_energy)

    # -----------------
    energy_differences, n_dofs, n_iterations, resolved = get_energy_diffs(
        path_to_results=path_to_results,
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

    # plot on log-log axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # connect points with a dotted black line (log-log)
    ax.loglog(
        n_dofs,
        energy_differences,
        linestyle=':',
        color='black',
        zorder=1
    )

    # plot resolved points (blue) and non-resolved points (red)
    resolved_mask = resolved.astype(bool)
    ax.scatter(
        n_dofs[resolved_mask],
        energy_differences[resolved_mask],
        color='blue',
        label='resolved',
        zorder=2,
        marker='o'
    )
    ax.scatter(
        n_dofs[~resolved_mask],
        energy_differences[~resolved_mask],
        color='red',
        label='not resolved',
        zorder=2,
        marker='o'
    )
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
    resolved = []

    for path_to_n_dofs in tqdm(list(path_to_results.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue 
        n_dof = int(path_to_n_dofs.name)
        n_dofs.append(n_dof)

        current_energy = load_dump(
            path_to_dump=path_to_n_dofs / 'energy.pkl')
        current_n_iterations = load_dump(
            path_to_dump=path_to_n_dofs / 'n_iterations.pkl')
        current_resolved = load_dump(
            path_to_dump=path_to_n_dofs / 'criterion_resolved.pkl')
        
        energies.append(current_energy)
        n_iterations.append(current_n_iterations)
        resolved.append(current_resolved)

    # converting lists to numpy arrays
    energies = np.array(energies)
    n_dofs = np.array(n_dofs)
    n_iterations = np.array(n_iterations)
    resolved = np.array(resolved)
    
    # sorting corresponding to number of degrees of freedom
    sort_n_dof = n_dofs.argsort()
    n_dofs = n_dofs[sort_n_dof]
    energies = energies[sort_n_dof]
    n_iterations = n_iterations[sort_n_dof]
    resolved = resolved[sort_n_dof]

    energy_differences = energies - reference_energy
    if np.any(energy_differences < 0.0):
        raise RuntimeError(
            "None of the energy differences should be negative!" \
            "This probably means that the reference mesh was too " \
            "coarse or the reference solution too rough"
        )

    return energy_differences, n_dofs, n_iterations, resolved


if __name__ == '__main__':
    main()
