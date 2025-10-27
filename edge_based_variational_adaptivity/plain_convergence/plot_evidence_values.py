from cProfile import label
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

    n_dofs: list[int] = []
    energies: list[float] = []
    energy_differences: list[float] = []
    sums_energy_decays: list[float] = []
    sums_marked_energy_decays: list[float] = []
    
    for n_dof_dir in base_path.iterdir():
        if not n_dof_dir.is_dir():
            continue

        n_dofs.append(int(n_dof_dir.name))

        path_to_energy_galerkin = n_dof_dir / Path('galerkin_solution_energy.pkl')
        energy_galerkin = load_dump(path_to_dump=path_to_energy_galerkin)
        energies.append(energy_galerkin)
        energy_differences.append(energy_galerkin - energy_ref)

        path_to_sum_energy_decays = n_dof_dir / Path('sum_energy_decays.pkl')
        path_to_sum_marked_energy_decays = n_dof_dir / Path('sum_marked_energy_decays.pkl')

        sum_marked_energy_decays = load_dump(path_to_dump=path_to_sum_marked_energy_decays)
        sum_energy_decays = load_dump(path_to_dump=path_to_sum_energy_decays)

        sums_energy_decays.append(sum_energy_decays)
        sums_marked_energy_decays.append(sum_marked_energy_decays)
    
    n_dofs = np.array(n_dofs)
    energy_differences = np.array(energy_differences)
    sums_energy_decays = np.array(sums_energy_decays)
    sums_marked_energy_decays = np.array(sums_marked_energy_decays)
    energies = np.array(energies)

    # sorting according to DOFs
    sort_n_dof = n_dofs.argsort()
    n_dofs = n_dofs[sort_n_dof]
    energy_differences = energy_differences[sort_n_dof]
    sums_energy_decays = sums_energy_decays[sort_n_dof]
    sums_marked_energy_decays = sums_marked_energy_decays[sort_n_dof]
    energies = energies[sort_n_dof]

    # extracting the relevant values
    # ------------------------------

    # (i) A posteriori error estimate
    # E(u_H) - E(u) < C_1 sum_{S in S_H} dE_S
    indicatori_i = energy_differences / sums_energy_decays
    label_i = r'$\frac{E(u_H) - E(u)}{\sum_{S \in S_H} \Delta E_S}$'

    # (ii) Lower bound on energy difference
    # sum_{S in \tilde S_H} dE_S < C_2 (E(u_H) - E(u_h))
    indicator_ii = sums_marked_energy_decays[:-1] / (energies[:-1] - energies[1:])
    label_ii = r'$\frac{\sum_{S \in \tilde S_H \Delta E_S}}{E(u_H) - E(u_h)}$'

    # (iii) Energy Contraction, i.e.
    # E(u_h) - E(u) < q [E(u_H) - E(u)],  q in (0, 1)
    energy_quotients = energy_differences[1:] / energy_differences[:-1]
    label_iii = r'$\frac{E(u_h) - E(u)}{E(u_H) - E(u)}$'

    plt.semilogx(n_dofs[:-1], energy_quotients, label=label_iii)
    plt.semilogx(n_dofs, indicatori_i, label=label_i)
    plt.semilogx(n_dofs[:-1], indicator_ii, label=label_ii)
    plt.hlines(y=1.0, xmin=n_dofs[2], xmax=n_dofs[-2], color='red')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
