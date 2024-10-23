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
from scipy.sparse.linalg import cg
from copy import copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--fudge", type=float, required=True,
                        help="if dE_n < fudge * sum_{k=1}^n dE_k/n, then"
                        " CG on current mesh is stopped and VA kicks in")
    args = parser.parse_args()

    THETA = args.theta
    FUDGE = args.fudge

    max_n_updates = 1000
    max_n_loops = 10

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_14') /
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

    # initializing the solution to random values
    current_iterate = np.random.rand(coordinates.shape[0])
    # forcing the boundary values to be zero, nevertheless
    current_iterate[np.unique(boundaries[0].flatten())] = 0.

    # ------------------------------------------------
    # variational adaptivity + Local Solvers
    # ------------------------------------------------

    # number of refinement steps using variational adaptivity

    for _ in range(max_n_loops):
        # it may be that the mesh has changed
        # this fact needs to be take into account
        # hence, we re-setup everything
        # ---------------------------------------
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

        # dump snapshot of current mesh state
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))

        # ------------------------------
        # Perform CG on the current mesh
        # ------------------------------
        current_iterate_cg = np.zeros(n_vertices)

        print('performing n global CG steps on current mesh')
        current_iterate_cg[free_nodes], _ = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes],
            maxiter=n_cg_steps,
            rtol=1e-100)

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


class StoppingException(Exception):
    """Exception to raise when custom stopping criterion is met."""
    current_iterate: np.ndarray

    def __init__(self, current_iterate, *args: object) -> None:
        super().__init__(*args)
        self.current_iterate = current_iterate


class FixedNumberOfStepsCallback():
    n_steps_max: int
    n_steps_performed: int

    def __init__(self, n_steps_max) -> None:
        self.n_steps_max = n_steps_max
        self.n_steps_performed = 0

    def __call__(self, current_iterate) -> None:
        # we know that scipy.sparse.linalg.cg calls this after each iteration
        self.n_steps_performed += 1

        # are we done?
        if self.n_steps_performed == self.n_steps_max:
            raise StoppingException(current_iterate=current_iterate)


class CustomCallBack():
    n_iterations_done: int
    accumulated_energy_drop: float
    n_dofs: int
    c: float
    base_results_path: Path
    free_nodes: np.ndarray
    exact_solution_full: np.ndarray
    stiffness_matrix_full: csr_matrix
    old_iterate: np.ndarray

    def __init__(
            self,
            n_dofs: int,
            c: float,
            base_results_path: Path,
            free_nodes: np.ndarray,
            stiffness_matrix: csr_matrix,
            right_hand_side: np.ndarray,
            initial_guess: np.ndarray) -> None:
        self.n_iterations_done = 0
        self.accumulated_energy_drop = 0
        self.n_dofs = n_dofs
        self.c = c
        self.base_results_path = base_results_path
        self.free_nodes = free_nodes
        self.stiffness_matrix = stiffness_matrix
        self.right_hand_side = right_hand_side
        self.old_iterate = initial_guess

    def energy(self, current_iterate) -> float:
        return 0.5 * (
            current_iterate.dot(self.stiffness_matrix.dot(
                current_iterate))
        ) - self.right_hand_side.dot(current_iterate)

    def get_energy_drop(self, old_iterate, current_iterate) -> float:
        old_energy = self.energy(old_iterate)
        new_energy = self.energy(current_iterate)
        return old_energy - new_energy

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

        energy_drop = self.get_energy_drop(
            old_iterate=self.old_iterate,
            current_iterate=current_iterate)

        self.accumulated_energy_drop += energy_drop

        self.old_iterate = copy(current_iterate)

        # are we done?
        mean_energy_drop = self.accumulated_energy_drop \
            / self.n_iterations_done
        has_converged = energy_drop < self.c * mean_energy_drop

        if has_converged:
            raise ConvergenceException(converged_iterate=current_iterate_full)


if __name__ == '__main__':
    main()
