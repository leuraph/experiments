import argparse
from pathlib import Path
from tqdm import tqdm
from load_save_dumps import load_dump
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    output_path = Path('plots') / Path(f'{args.path}.pdf').name

    path_to_results = Path(args.path)

    energy_norm_errors_squared =[]
    n_dofs = []

    for path_to_n_dofs in tqdm(list(path_to_results.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue 
        n_dof = int(path_to_n_dofs.name)
        n_dofs.append(n_dof)

        current_energy = load_dump(
            path_to_dump=path_to_n_dofs / 'energy_norm_error_squared.pkl')

        energy_norm_errors_squared.append(current_energy)

    # converting lists to numpy arrays
    energy_norm_errors_squared = np.array(energy_norm_errors_squared)
    n_dofs = np.array(n_dofs)
    
    # sorting corresponding to number of degrees of freedom
    sort_n_dof = n_dofs.argsort()
    n_dofs = n_dofs[sort_n_dof]
    energy_norm_errors_squared = energy_norm_errors_squared[sort_n_dof]

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
    ax.set_ylabel(r'$E(\tilde u) - E(u)$')
    ax.grid(True)

    ax.loglog(
            n_dofs,
            energy_norm_errors_squared,
            linestyle='--')

    # ax.legend(loc='best')

    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()

