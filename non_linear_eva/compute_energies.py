import argparse
from pathlib import Path
from tqdm import tqdm
from load_save_dumps import dump_object, load_dump
import numpy as np
from scipy.sparse import csr_matrix
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side, \
    integrate_composition_nonlinear_with_fem, get_mass_matrix, get_general_stiffness_matrix
from p1afempy.data_structures import ElementsType, CoordinatesType
from triangle_cubature.cubature_rule import CubatureRuleEnum
from problems import get_problem, Problem

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    problem = get_problem(number=args.problem)

    path_to_results = Path(args.path)

    for path_to_n_dofs in tqdm(list(path_to_results.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue 

        path_to_results_coordinates = path_to_n_dofs / Path("coordinates.pkl")
        path_to_results_elements = path_to_n_dofs / Path("elements.pkl")
        path_to_results_solution = path_to_n_dofs / Path("last_iterate.pkl")

        coordinates = load_dump(path_to_dump=path_to_results_coordinates)
        elements = load_dump(path_to_dump=path_to_results_elements)
        last_iterate = load_dump(path_to_dump=path_to_results_solution)

        energy = compute_energy(
            problem=problem,
            elements=elements,
            coordinates=coordinates,
            current_iterate=last_iterate)
        
        path_energy = path_to_n_dofs / "energy.pkl"
        
        dump_object(
            obj=energy,
            path_to_file=path_energy
        )


def compute_energy(
        problem: Problem,
        elements: ElementsType,
        coordinates: CoordinatesType,
        current_iterate: np.ndarray
        ) -> float:
    """
    given a problem, computes the energy
    E(u) := 1/2 \int <A(x) nabla u(x), nabla u(x)> + int Phi(u) - int fu
    """
    stiffness_matrix = csr_matrix(
        get_general_stiffness_matrix(
            elements=elements,
            coordinates=coordinates,
            a_11=problem.a_11,
            a_12=problem.a_12,
            a_21=problem.a_21,
            a_22=problem.a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR))
    right_hand_side_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=problem.f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    energy = (
            0.5 * current_iterate.dot(stiffness_matrix.dot(current_iterate))
            +
            integrate_composition_nonlinear_with_fem(
                f=problem.Phi,
                u=current_iterate,
                coordinates=coordinates,
                elements=elements,
                cubature_rule=CubatureRuleEnum.DAYTAYLOR)
            -
            right_hand_side_vector.dot(current_iterate)
        )
    return energy



def upper_bound(
        reference_solution: np.ndarray,
        results_solution: np.ndarray,
        reference_elements: ElementsType,
        reference_coordinates: CoordinatesType,
        results_elements: ElementsType,
        results_coordinates: CoordinatesType,
        phi, f) -> float:
    """
    based on the computable upper bound
    |u - u_tilde|^2_a
    \leq
    a(u,u) + a(u_tilde, u_tilde) - 2 <f, u_tilde> + 2 |phi(u)|_L2 |u_tilde|_L2

    note
    ----
    this is just an approximative upper bound as,
    in order to derive it, we used that u is indeed the exact solution.
    in this implementation, however, we would
    use the reference solution on the graded mesh
    """
    stiffness_matrix_reference = csr_matrix(get_stiffness_matrix(
        coordinates=reference_coordinates,
        elements=reference_elements
    ))
    auu = reference_solution.dot(
        stiffness_matrix_reference.dot(reference_solution))

    stiffness_matrix_results = csr_matrix(get_stiffness_matrix(
        coordinates=results_coordinates,
        elements=results_elements
    ))
    

    mass_matrix_reference = csr_matrix(get_mass_matrix(
        coordinates=reference_coordinates,
        elements=reference_elements
    ))
    right_hand_silde_results = get_right_hand_side(
        coordinates=results_coordinates,
        elements=results_elements,
        f=f, cubature_rule=CubatureRuleEnum.DAYTAYLOR
    )
    phiu_norm = np.sqrt(integrate_composition_nonlinear_with_fem(
        f=lambda x: phi(x)**2,
        u=results_solution,
        coordinates=results_coordinates,
        elements=results_elements,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR
    ))
    u_tilde_norm = np.sqrt(
        reference_solution.dot(mass_matrix_reference.dot(reference_solution)))
    
    return (
        reference_solution.dot(
            stiffness_matrix_reference.dot(reference_solution))
        +
        results_solution.dot(
            stiffness_matrix_results.dot(results_solution))
        -
        2. * right_hand_silde_results.dot(results_solution)
        +
        2. * phiu_norm * u_tilde_norm
    )


if __name__ == '__main__':
    main()

