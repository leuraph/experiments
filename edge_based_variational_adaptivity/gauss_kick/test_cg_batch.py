import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import random
from scipy.sparse.linalg import spsolve
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.refinement import refineNVB, refine_single_edge, refineNVB_edge_based
from p1afempy.mesh import provide_geometric_data, get_local_patch_edge_based, show_mesh
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side
from pathlib import Path
from configuration import f, uD
from load_save_dumps import dump_object
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
from scipy.sparse.linalg import cg
from copy import copy
from ismember import is_row_in
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from p1afempy.data_structures import ElementsType, CoordinatesType


def main() -> None:
    np.random.seed(42)

    n_initial_refinements = 6
    cg_steps_in_batch = 5
    max_n_batch_solves = 20
    max_iter = max_n_batch_solves * cg_steps_in_batch

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements_order1.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    # initial refinement
    # ------------------
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)

    stiffness_matrix = csr_matrix(get_stiffness_matrix(
        coordinates=coordinates, elements=elements))
    right_hand_side = get_right_hand_side(
        coordinates=coordinates, elements=elements, f=f)

    # initializing the solution to random values
    initial_guess = np.random.rand(coordinates.shape[0])
    # forcing the boundary values to be zero, nevertheless
    initial_guess[np.unique(boundaries[0].flatten())] = 0.

    # calculating free nodes on the initial mesh
    # ------------------------------------------
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    # initial exact galerkin solution
    # -------------------------------
    solution, _ = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f,
        g=None,
        uD=uD)

    def get_energy_error_squared(
            current_iterate_on_free_nodes: np.ndarray) -> float:
        du = solution[free_nodes] - current_iterate_on_free_nodes
        return du.dot(stiffness_matrix[free_nodes, :][:, free_nodes].dot(du))

    def get_energy(current_iterate_on_free_nodes: np.ndarray) -> float:
        return (
            0.5 * (
                current_iterate_on_free_nodes.dot(
                    stiffness_matrix[free_nodes, :][:, free_nodes].dot(
                        current_iterate_on_free_nodes)))
            - right_hand_side[free_nodes].dot(current_iterate_on_free_nodes))

    _, edge_to_nodes, _ = provide_geometric_data(
        elements=elements,
        boundaries=boundaries)

    # CG with interruptions
    # ---------------------
    current_iterate = copy(initial_guess)

    energy_errors_squared_with_interruption = [
        get_energy_error_squared(
            current_iterate_on_free_nodes=current_iterate[free_nodes])]
    energies_with_interruptions = [
        get_energy(
            current_iterate_on_free_nodes=current_iterate[free_nodes])]
    energy_after_eva = get_energy_after_eva(
        edge_to_nodes=edge_to_nodes,
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries,
        current_iterate=current_iterate,
        f=f)
    initial_energy = get_energy(
            current_iterate_on_free_nodes=current_iterate[free_nodes])
    eva_energy_losses_with_interruptions = [
        initial_energy - energy_after_eva]

    for _ in range(max_n_batch_solves):
        current_iterate[free_nodes], info = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes], rtol=1e-100,
            maxiter=cg_steps_in_batch)
        energy_errors_squared_with_interruption.append(
            get_energy_error_squared(
                current_iterate_on_free_nodes=current_iterate[free_nodes]))
        energies_with_interruptions.append(
            get_energy(
                current_iterate_on_free_nodes=current_iterate[free_nodes]))
        energy_after_eva = get_energy_after_eva(
            edge_to_nodes=edge_to_nodes,
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=current_iterate,
            f=f)
        eva_energy_losses_with_interruptions.append(
            energies_with_interruptions[-1] - energy_after_eva)

    # CG without interruptions
    # ------------------------
    current_iterate = copy(initial_guess)

    energy_errors_squared_without_interruption = [
        get_energy_error_squared(
            current_iterate_on_free_nodes=current_iterate[free_nodes])]
    energies_without_interruptions = [
        get_energy(
            current_iterate_on_free_nodes=current_iterate[free_nodes])]

    energy_after_eva = get_energy_after_eva(
        edge_to_nodes=edge_to_nodes,
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries,
        current_iterate=current_iterate,
        f=f)
    initial_energy = get_energy(
            current_iterate_on_free_nodes=current_iterate[free_nodes])
    eva_energy_losses_without_interruptions = [
        initial_energy - energy_after_eva]

    custom_callback = CustomCallBack(
        cg_steps_in_batch=cg_steps_in_batch,
        lamba=get_energy_error_squared,
        get_energy=get_energy,
        get_energy_error_squared=get_energy_error_squared,
        energies=energies_without_interruptions,
        energy_errors_squared=energy_errors_squared_without_interruption,
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries,
        eva_energy_losses=eva_energy_losses_without_interruptions,
        free_nodes=free_nodes)
    current_iterate, info = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes], rtol=1e-100,
            maxiter=max_iter,
            callback=custom_callback)
    energy_errors_squared_without_interruption = \
        custom_callback.energy_errors_squared

    print(len(energy_errors_squared_with_interruption))
    print(len(energy_errors_squared_without_interruption))

    fig, ax = plt.subplots(3, 1)

    # energy errors plot
    # ------------------
    ax[0].semilogy(
        energy_errors_squared_with_interruption,
        label='with interruptions')
    ax[0].semilogy(
        energy_errors_squared_without_interruption,
        label='without interruptions')
    ax[0].legend(loc='best')
    ax[0].grid(True)
    ax[0].set_title(r'$\|u_N^n - u_h\|_a^2$')

    energies_without_interruptions = np.array(
        energies_without_interruptions)
    energies_with_interruptions = np.array(
        energies_with_interruptions)
    energy_losses_without_iterruptions = (
        energies_without_interruptions[:-1] -
        energies_without_interruptions[1:])
    energy_losses_with_iterruptions = (
        energies_with_interruptions[:-1] -
        energies_with_interruptions[1:])
    cum_energy_losses_with_interruptions =\
        np.cumsum(energy_losses_with_iterruptions)
    cum_energy_losses_without_interruptions =\
        np.cumsum(energy_losses_without_iterruptions)

    # cummulative energy loss plot
    # ----------------------------
    ax[1].semilogy(
        np.abs(cum_energy_losses_with_interruptions),
        label='with interruptions')
    ax[1].semilogy(
        np.abs(cum_energy_losses_without_interruptions),
        label='without interruptions')
    ax[1].legend(loc='best')
    ax[1].grid(True)
    ax[1].set_title(r'$E (u_0) - E(u_{n})$')

    # energy losses per batch plot
    # ----------------------------
    ax[2].semilogy(
        np.abs(energy_losses_with_iterruptions),
        label='with interruptions')
    ax[2].semilogy(
        np.abs(energy_losses_without_iterruptions),
        label='without interruptions')
    ax[2].semilogy(
        np.abs(custom_callback.eva_energy_losses[:-1]),
        label='EVA without interruptions')
    ax[2].semilogy(
        np.abs(eva_energy_losses_with_interruptions[:-1]),
        label='EVA with interruptions')
    ax[2].legend(loc='best')
    ax[2].grid(True)
    ax[2].set_title(r'$E (u_n) - E(u_{n+1})$')

    # --------
    # plotting
    # --------
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 16

    # Add a main title for the entire figure
    fig.suptitle(
        f"{n_initial_refinements} initial refinements, "
        + f"{cg_steps_in_batch} cg steps in one batch",
        fontsize=12)

    fig.set_size_inches(16, 12)
    fig.savefig(
        'plots/cg_vs_eva.pdf',
        dpi=300, bbox_inches="tight")


class CustomCallBack():
    n_iterations_done: int
    cg_steps_in_batch: int
    energy_errors_squared: list[float]
    energies: list[float]
    eva_energy_losses: list[float]

    def __init__(
            self, cg_steps_in_batch, lamba,
            energy_errors_squared,
            energies,
            eva_energy_losses,
            get_energy_error_squared,
            get_energy,
            coordinates,
            elements,
            boundaries,
            free_nodes) -> None:
        self.n_iterations_done = 0
        self.cg_steps_in_batch = cg_steps_in_batch
        self.energy_errors_squared = energy_errors_squared
        self.energies = energies
        self.eva_energy_losses = eva_energy_losses
        self.lamba = lamba
        self.get_energy_error_squared = get_energy_error_squared
        self.get_energy = get_energy
        # mesh
        self.coordinates = coordinates
        self.elements = elements
        self.boundaries = boundaries
        self.free_nodes = free_nodes

    def __call__(self, current_iterate_on_free_nodes):
        # we know that scipy.sparse.linalg.cg calls this after each iteration
        self.n_iterations_done += 1

        if self.n_iterations_done % self.cg_steps_in_batch == 0:
            self.energy_errors_squared.append(
                self.lamba(current_iterate_on_free_nodes))
            self.energies.append(
                self.get_energy(current_iterate_on_free_nodes))

            # --------------------------------------
            # perform EVA with old_iterate -> dE_EVA
            # --------------------------------------
            element_to_edges, edge_to_nodes, boundaries_to_edges = \
                provide_geometric_data(
                    elements=self.elements,
                    boundaries=self.boundaries)

            n_boundaries = edge_to_nodes.shape[0]

            edge_to_nodes_flipped = np.column_stack(
                [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
            boundary = np.logical_or(
                is_row_in(edge_to_nodes, self.boundaries[0]),
                is_row_in(edge_to_nodes_flipped, self.boundaries[0])
            )
            non_boundary = np.logical_not(boundary)
            non_boundary_edges = edge_to_nodes[non_boundary]

            # we get a new value for each new edge
            values_on_new_edges = np.zeros(n_boundaries)

            # restoring the full current iterate for
            # convenience when applying EVA
            current_iterate = np.zeros(self.coordinates.shape[0])
            current_iterate[self.free_nodes] = current_iterate_on_free_nodes

            right_hand_side = get_right_hand_side(
                coordinates=self.coordinates,
                elements=self.elements,
                f=f)
            stiffness_matrix = csr_matrix(get_stiffness_matrix(
                coordinates=self.coordinates,
                elements=self.elements))

            right_hand_side_on_free_nodes = right_hand_side[self.free_nodes]
            stiffness_matrix_on_free_nodes = stiffness_matrix[
                self.free_nodes, :][:, self.free_nodes]

            # computing global terms before loop
            L_1 = right_hand_side_on_free_nodes.dot(
                current_iterate_on_free_nodes)
            A_11 = current_iterate_on_free_nodes.dot(
                stiffness_matrix_on_free_nodes.dot(
                    current_iterate_on_free_nodes))

            # we get a new value for each new edge
            n_non_boundary_edges = non_boundary_edges.shape[0]
            values_on_new_edges_non_boundary = np.zeros(n_non_boundary_edges)
            energy_gains = np.zeros(n_non_boundary_edges)

            for k, non_boundary_edge in enumerate(tqdm(non_boundary_edges)):

                local_elements, local_coordinates, \
                    local_iterate, local_edge_indices = \
                    get_local_patch_edge_based(
                        elements=self.elements,
                        coordinates=self.coordinates,
                        current_iterate=current_iterate,
                        edge=non_boundary_edge)
                tmp_local_coordinates, tmp_local_elements, \
                    tmp_local_solution =\
                    refine_single_edge(
                        coordinates=local_coordinates,
                        elements=local_elements,
                        edge=local_edge_indices,
                        to_embed=local_iterate)
                tmp_stiffness_matrix = csr_matrix(get_stiffness_matrix(
                    coordinates=tmp_local_coordinates,
                    elements=tmp_local_elements))
                tmp_rhs_vector = get_right_hand_side(
                    coordinates=tmp_local_coordinates,
                    elements=tmp_local_elements, f=f)

                # building the local 2x2 system
                A_12 = tmp_stiffness_matrix.dot(tmp_local_solution)[-1]
                A_22 = tmp_stiffness_matrix[-1, -1]

                L_2 = tmp_rhs_vector[-1]

                detA = (A_11 * A_22 - A_12 * A_12)

                alpha = (A_22 * L_1 - A_12 * L_2)/detA
                beta = (-A_12 * L_1 + A_11 * L_2)/detA

                dE = 0.5*(
                    (alpha-1)**2 * A_11
                    + 2.*(alpha-1)*beta*A_12
                    + beta**2 * A_22)

                energy_gains[k] = dE
                i, j = local_edge_indices
                values_on_new_edges_non_boundary[k] = beta + 0.5 * (
                    local_iterate[i] + local_iterate[j])

            values_on_new_edges[non_boundary] = \
                values_on_new_edges_non_boundary

            current_iterate_after_eva = np.hstack(
                [current_iterate, values_on_new_edges])

            # mark all elements for refinement
            marked_elements = np.arange(self.elements.shape[0])
            new_coordinates, new_elements, _, _ =\
                refineNVB(
                    coordinates=self.coordinates,
                    elements=self.elements,
                    marked_elements=marked_elements,
                    boundary_conditions=self.boundaries)

            new_stiffness_matrix = csr_matrix(
                get_stiffness_matrix(
                    coordinates=new_coordinates,
                    elements=new_elements))
            new_right_hand_side = get_right_hand_side(
                coordinates=new_coordinates, elements=new_elements, f=f)

            eva_energy = 0.5 * current_iterate_after_eva.dot(
                new_stiffness_matrix.dot(current_iterate_after_eva)) \
                - new_right_hand_side.dot(current_iterate_after_eva)

            dE_eva = self.energies[-1] - eva_energy
            self.eva_energy_losses.append(dE_eva)


def get_energy_after_eva(
        edge_to_nodes,
        coordinates,
        elements,
        boundaries,
        current_iterate,
        f) -> float:
    """
    given the current iterate, calculates the possible energy drop
    using edge based variational adaptivity (EVA)
    """

    n_boundaries = edge_to_nodes.shape[0]

    edge_to_nodes_flipped = np.column_stack(
        [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
    boundary = np.logical_or(
        is_row_in(edge_to_nodes, boundaries[0]),
        is_row_in(edge_to_nodes_flipped, boundaries[0])
    )
    non_boundary = np.logical_not(boundary)
    non_boundary_edges = edge_to_nodes[non_boundary]

    # we get a new value for each new edge
    values_on_new_edges = np.zeros(n_boundaries)

    right_hand_side = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f)
    stiffness_matrix = csr_matrix(get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements))

    # computing global terms before loop
    L_1 = right_hand_side.dot(
        current_iterate)
    A_11 = current_iterate.dot(
        stiffness_matrix.dot(
            current_iterate))

    # we get a new value for each new edge
    n_non_boundary_edges = non_boundary_edges.shape[0]
    values_on_new_edges_non_boundary = np.zeros(n_non_boundary_edges)

    for k, non_boundary_edge in enumerate(tqdm(non_boundary_edges)):

        local_elements, local_coordinates, \
            local_iterate, local_edge_indices = \
            get_local_patch_edge_based(
                elements=elements,
                coordinates=coordinates,
                current_iterate=current_iterate,
                edge=non_boundary_edge)
        tmp_local_coordinates, tmp_local_elements, \
            tmp_local_solution =\
            refine_single_edge(
                coordinates=local_coordinates,
                elements=local_elements,
                edge=local_edge_indices,
                to_embed=local_iterate)
        tmp_stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=tmp_local_coordinates,
            elements=tmp_local_elements))
        tmp_rhs_vector = get_right_hand_side(
            coordinates=tmp_local_coordinates,
            elements=tmp_local_elements, f=f)

        # building the local 2x2 system
        A_12 = tmp_stiffness_matrix.dot(tmp_local_solution)[-1]
        A_22 = tmp_stiffness_matrix[-1, -1]

        L_2 = tmp_rhs_vector[-1]

        detA = (A_11 * A_22 - A_12 * A_12)

        alpha = (A_22 * L_1 - A_12 * L_2)/detA
        beta = (-A_12 * L_1 + A_11 * L_2)/detA

        i, j = local_edge_indices
        values_on_new_edges_non_boundary[k] = beta + 0.5 * (
            local_iterate[i] + local_iterate[j])

    values_on_new_edges[non_boundary] = \
        values_on_new_edges_non_boundary

    current_iterate_after_eva = np.hstack(
        [current_iterate, values_on_new_edges])

    # mark all elements for refinement
    marked_elements = np.arange(elements.shape[0])
    new_coordinates, new_elements, _, _ =\
        refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)

    new_stiffness_matrix = csr_matrix(
        get_stiffness_matrix(
            coordinates=new_coordinates,
            elements=new_elements))
    new_right_hand_side = get_right_hand_side(
        coordinates=new_coordinates, elements=new_elements, f=f)

    eva_energy = 0.5 * current_iterate_after_eva.dot(
        new_stiffness_matrix.dot(current_iterate_after_eva)) \
        - new_right_hand_side.dot(current_iterate_after_eva)

    return eva_energy


if __name__ == '__main__':
    main()
