import matplotlib.pyplot as plt
from pathlib import Path
from load_save_dumps import load_dump
import numpy as np
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, required=True,
                        help="path to the output dir")
    parser.add_argument("--theta", type=float, required=False)
    args = parser.parse_args()
    theta = args.theta
    path_to_output_dir = Path(args.o)

    solver_names = [
        'local_jacobi',
        'local_block_jacobi',
        'local_gauss_seidel',
        'local_context_solver_non_simultaneous',
        'local_context_solver_simultaneous']

    # loop over solver
    for solver_name in solver_names:
        # get all three results for fixed theta
        base_results_path: list[Path] = [
            Path(f'results/1/{solver_name}/energy_norm_errors'),
            Path(f'results/2/{theta}/{solver_name}/energy_norm_errors'),
            Path(f'results/3/{theta}/{solver_name}/energy_norm_errors')
        ]

        # ------------
        # read results
        # ------------
        energy_norm_errors_squared = []
        n_local_solves = []

        for base_result_path in base_results_path:
            # looping over iterates
            errs = []
            nsolvs = []
            for path_to_energy_norm_error in base_result_path.iterdir():
                nsolvs.append(int(path_to_energy_norm_error.stem))
                errs.append(load_dump(path_to_dump=path_to_energy_norm_error))
            energy_norm_errors_squared.append(errs)
            n_local_solves.append(nsolvs)

        energy_norm_errors_squared = np.array(energy_norm_errors_squared)
        n_local_solves = np.array(n_local_solves)

        # --------
        # plotting
        # --------
        path_to_plot = path_to_output_dir / Path(
            f'energy_norm_errors_{solver_name}_theta-{theta}.pdf')
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['legend.fontsize'] = 16
        plt.rcParams["figure.figsize"] = [6.4*1.5, 4.8*1.5]
        merged = []
        labels = []

        # https://coolors.co/palette/540d6e-ee4266-ffd23f-3bceac-0ead69
        COLORS = [
            '#540D6E',
            '#ee4266',
            '#FFD23F']
            # '#0EAD69',
            #'#3BCEAC']

        fig, ax = plt.subplots()
        ax.set_xlabel(r'$n_{\text{local solves}}$')
        ax.set_ylabel(r'$\| u_n - u_h \|_a^2$')
        ax.grid(True)

        for errs, nsolvs, exp_n, color in zip(
                energy_norm_errors_squared, n_local_solves, [1, 2, 3], COLORS):
            sort_indices = nsolvs.argsort()

            errs = errs[sort_indices]
            nsolvs = nsolvs[sort_indices]

            line, = ax.loglog(
                nsolvs, errs, '--', linewidth=1.2, alpha=1, color=color)
            mark, = ax.loglog(
                nsolvs, errs, linestyle=None, marker='s',
                markersize=8, linewidth=0, alpha=0.6, color=color)
            merged.append((line, mark))
            labels.append(f'experimet nr. {exp_n}')

        ax.legend(merged, labels)
        fig.savefig(
            path_to_plot,
            dpi=300, bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    main()
