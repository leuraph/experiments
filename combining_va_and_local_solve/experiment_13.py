import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import get_element_to_neighbours
from pathlib import Path
from variational_adaptivity import algo_4_1
from experiment_setup import f, uD
from load_save_dumps import dump_object
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--fudge", type=float, required=True,
                        help="if sum_{k=1}^n dE_k/n > fudge * dE_n, then"
                        " VA kicks in")
    args = parser.parse_args()

    THETA = args.theta
    FUDGE = args.fudge

    max_n_updates = 1000
    min_n_updates = 10

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_10') /
        Path(f'theta-{THETA}_fudge-{FUDGE}'))

    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    # initial refinement
    # ------------------
    n_initial_refinements = 3
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)

    # ------------------------------------------------
    # variational adaptivity + Local Solvers
    # ------------------------------------------------

    # initializing the solution to random values
    current_iterate = np.random.rand(coordinates.shape[0])
    # forcing the boundary values to be zero, nevertheless
    current_iterate[np.unique(boundaries[0].flatten())] = 0.

    # number of refinement steps using variational adaptivity
    n_va_refinement_steps = 8
    for _ in range(n_va_refinement_steps):
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        # assembly of right hand side
        right_hand_side = solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f)

        # assembly of the stiffness matrix
        stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))

        # -----------------------------------
        # compute and drop the exact solution
        # -----------------------------------
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f,
            g=None,
            uD=uD)

        dump_object(
            obj=solution, path_to_file=base_results_path /
            Path(f'{n_dofs}/exact_solution.pkl'))

        # ------------------------------------------------------------
        # compute all energy gains / local increments via local solver
        # ------------------------------------------------------------
        print('performing global update steps')

        accumulated_energy_drop = 0
        for n_update in tqdm.tqdm(range(max_n_updates)):

            # ----------------------------------------------------
            # Perform a global line search in gradient's direction
            # ----------------------------------------------------
            residual_free_nodes = (
                right_hand_side[free_nodes]
                - stiffness_matrix[free_nodes, :][:, free_nodes].dot(
                    current_iterate[free_nodes]))

            # step size calculation
            numerator = residual_free_nodes.dot(residual_free_nodes)
            denominator = residual_free_nodes.dot(
                stiffness_matrix[free_nodes, :][:, free_nodes].dot(
                    residual_free_nodes))
            step_size = numerator / denominator

            def energy(iterate) -> float:
                return (
                    0.5 * (iterate[free_nodes].dot(
                        stiffness_matrix[free_nodes, :][:, free_nodes].dot(
                            iterate[free_nodes]
                        )))
                    - right_hand_side[free_nodes].dot(
                        iterate[free_nodes]))

            current_energy = energy(current_iterate)

            # perform update
            global_free_increment = step_size * residual_free_nodes
            current_iterate[free_nodes] += global_free_increment

            new_energy = energy(current_iterate)

            energy_drop = current_energy - new_energy
            accumulated_energy_drop += energy_drop

            # dump snapshot of current current state
            dump_object(obj=current_iterate, path_to_file=base_results_path /
                        Path(f'{n_dofs}/{n_update+1}/solution.pkl'))
            dump_object(obj=elements, path_to_file=base_results_path /
                        Path(f'{n_dofs}/elements.pkl'))
            dump_object(obj=coordinates, path_to_file=base_results_path /
                        Path(f'{n_dofs}/coordinates.pkl'))
            dump_object(obj=boundaries, path_to_file=base_results_path /
                        Path(f'{n_dofs}/boundaries.pkl'))

            if n_update + 1 < min_n_updates:
                continue

            if energy_drop <= FUDGE * accumulated_energy_drop/(n_update+1):
                break

        # --------------------------------------------------------------
        # compute all local energy gains via VA, based on exact solution
        # --------------------------------------------------------------
        element_to_neighbours = get_element_to_neighbours(elements=elements)
        print('computing all local energy gains with variational adaptivity')
        local_energy_differences_refine = algo_4_1.get_all_local_enery_gains(
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=solution,
            element_to_neighbours=element_to_neighbours,
            uD=uD,
            rhs_function=f, lamba_a=1)

        # -------------------------------------
        # refine elements marked for refinement
        # -------------------------------------
        marked = doerfler_marking(
            input=local_energy_differences_refine, theta=THETA)

        coordinates, elements, boundaries, current_iterate = \
            refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries,
                to_embed=current_iterate)


class ConvergenceException(Exception):
    """Exception to raise when custom convergence criterion is met."""
    converged_iterate: np.ndarray

    def __init__(self, converged_iterate, *args: object) -> None:
        super().__init__(*args)
        self.converged_iterate = converged_iterate


class CustomCallBack():
    n_iterations_done: int
    accumulated_energy_drop: float
    n_dofs: int
    c: float
    base_results_path: Path
    free_nodes: np.ndarray
    exact_solution_full: np.ndarray
    stiffness_matrix_full: csr_matrix

    def __init__(
            self,
            n_dofs: int,
            c: float,
            base_results_path: Path,
            free_nodes: np.ndarray,
            exact_solution_full: np.ndarray,
            stiffness_matrix_full: csr_matrix,
            right_hand_side_full: np.ndarray) -> None:
        self.n_iterations_done = 0
        self.accumulated_energy_drop = 0
        self.n_dofs = n_dofs
        self.c = c
        self.base_results_path = base_results_path
        self.free_nodes = free_nodes
        self.exact_solution_full = exact_solution_full
        self.stiffness_matrix_full = stiffness_matrix_full
        self.right_hand_side_full = right_hand_side_full

    def _has_converged(self, current_iterate_full) -> bool:
        return (
            self.energy_norm_error(current_iterate_full)
            <= self.c/self.n_dofs**0.5)

    def energy_norm_error(self, current_iterate_full) -> float:
        du = (self.exact_solution_full - current_iterate_full)
        return np.sqrt(du.dot(self.stiffness_matrix_full.dot(du)))

    def energy(self, current_iterate_full) -> float:
        return 0.5 * (
            current_iterate_full.dot(self.stiffness_matrix_full.dot(
                current_iterate_full))
        ) - self.right_hand_side_full.dot(current_iterate_full)

    def __call__(self, current_iterate):
        # we know that scipy.sparse.linalg.cg calls this after each iteration
        self.n_iterations_done += 1

        # prepare the full iterate for dumping
        current_iterate_full = np.zeros_like(self.free_nodes, dtype=float)
        current_iterate_full[self.free_nodes] = current_iterate

        # save the current iterate
        dump_object(
            obj=current_iterate_full,
            path_to_file=(
                self.base_results_path /
                Path(f'{self.n_dofs}/{self.n_iterations_done}/solution.pkl')))

        if self._has_converged(current_iterate_full):
            raise ConvergenceException(converged_iterate=current_iterate_full)



if __name__ == '__main__':
    main()
