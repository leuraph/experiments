import numpy as np
from problems import get_problem
from p1afempy import refinement
from p1afempy.solvers import \
    integrate_composition_nonlinear_with_fem, \
        get_load_vector_of_composition_nonlinear_with_fem, \
            get_general_stiffness_matrix, get_right_hand_side
from scipy.sparse import csr_matrix
from triangle_cubature.cubature_rule import CubatureRuleEnum
from ismember import is_row_in
from p1afempy.mesh import provide_geometric_data
from scipy.sparse.linalg import spsolve

def main() -> None:
    n_initial_refinements = 5
    max_dof = 1e6

    problem = get_problem(number=1)

    # extracting the initial mesh
    # ---------------------------
    initial_coarse_mesh = problem.get_coarse_initial_mesh()
    coordinates = initial_coarse_mesh.coordinates
    elements = initial_coarse_mesh.elements
    boundaries = initial_coarse_mesh.boundaries

    # extracting the corresponding PDE's data
    # ---------------------------------------
    f = problem.f
    phi = problem.phi
    phi_prime = problem.phi_prime
    Phi = problem.Phi
    a_11 = problem.a_11
    a_12 = problem.a_12
    a_21 = problem.a_21
    a_22 = problem.a_22
    # ---------------------------------------

    # initial refinement
    # ---------------------------------------
    for _ in range(n_initial_refinements):
        marked_elements = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)
    print(f'DOF = {n_dofs}')
    # ---------------------------------------

    # on the initial mesh, solve the auxiliary problem
    # -Laplace phi = 1, with homogeneous boundary conditions
    # ------------------------------------------------------
    
    auxiliary_problem = get_problem(number=2)

    f_aux = auxiliary_problem.f
    a_11_aux = auxiliary_problem.a_11
    a_12_aux = auxiliary_problem.a_12
    a_21_aux = auxiliary_problem.a_21
    a_22_aux = auxiliary_problem.a_22

    stiffness_matrix_aux = csr_matrix(get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=a_11_aux, a_12=a_12_aux, a_21=a_21_aux, a_22=a_22_aux,
        cubature_rule=CubatureRuleEnum.MIDPOINT))

    right_hand_side_vector_aux = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f_aux,
        cubature_rule=CubatureRuleEnum.MIDPOINT)
    
    n_coordinates = coordinates.shape[0]
    phi = np.zeros(n_coordinates)

    phi[free_nodes] = spsolve(
        stiffness_matrix_aux[free_nodes, :][:, free_nodes],
        right_hand_side_vector_aux[free_nodes],
        use_umfpack=True)
    
    psi_max = np.max(phi)

    lamba = 12.
    c = np.sqrt((lamba*psi_max - 1)/(psi_max**3))
    # ------------------------------------------------------

    initial_gues_plus = c*phi
    initial_gues_minus = -c*phi


if __name__ == '__main__':
    main()
