from pathlib import Path
from load_save_dumps import load_dump
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the results")
    args = parser.parse_args()

    base_path = Path(args.path)

    energy_norm_squared_ref = 0.214075802220546
    energy_ref = -0.5*energy_norm_squared_ref

    n_dofs = []
    energy_differences = []
    
    for n_dof_dir in base_path.iterdir():
        if not n_dof_dir.is_dir():
            continue

        n_dofs.append(int(n_dof_dir.name))

        path_to_energy_galerkin = n_dof_dir / Path('galerkin_solution_energy.pkl')
        energy_galerkin = load_dump(path_to_dump=path_to_energy_galerkin)
        energy_differences.append(energy_galerkin - energy_ref)
    
    n_dofs = np.array(n_dofs)
    energy_differences = np.array(energy_differences)

    # sorting according to DOFs
    sort_n_dof = n_dofs.argsort()
    n_dofs = n_dofs[sort_n_dof]
    energy_differences = energy_differences[sort_n_dof]

    energy_quotients = energy_differences[1:] / energy_differences[:-1]

    plt.semilogy(energy_quotients)
    plt.show()

if __name__ == '__main__':
    main()
