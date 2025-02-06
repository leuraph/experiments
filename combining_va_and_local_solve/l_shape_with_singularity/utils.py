import p1afempy
import numpy as np
from triangle_cubature.cubature_rule import CubatureRuleEnum
from triangle_cubature.rule_factory import get_rule


def get_energy_per_element(
        coordinates: p1afempy.data_structures.CoordinatesType,
        elements: p1afempy.data_structures.ElementsType,
        current_iterate: np.ndarray,
        cubature_rule: CubatureRuleEnum,
        f: p1afempy.data_structures.BoundaryConditionType
) -> np.ndarray:
    """
    Compute the energy contribution of each element in a P1 FEM triangulation.

    The energy is computed as:
        E_T = - ∫_T f u^n dx + 0.5 * ∫_T |∇u^n|^2 dx

    Parameters:
        coordinates (CoordinatesType):
            N_C x 2 array of (x, y) coordinates.
        elements (ElementsType):
            N_E x 3 array of node indices forming each triangle.
        current_iterate (np.ndarray):
            N_C array of function values at the nodes.
        cubature_rule (CubatureRuleEnum):
            Rule for numerical integration.
        f (BoundaryConditionType):
            Right-hand side function f(x, y).

    Returns:
        np.ndarray: N_E array of energy values for each element.
    """
    # Extract triangle vertex coordinates
    X = coordinates[elements, 0]  # Shape: (N_E, 3)
    Y = coordinates[elements, 1]  # Shape: (N_E, 3)

    z_0 = coordinates[elements[:, 0], :]
    z_1 = coordinates[elements[:, 1], :]
    z_2 = coordinates[elements[:, 2], :]

    # Compute edge vectors
    v1x, v1y = X[:, 1] - X[:, 0], Y[:, 1] - Y[:, 0]
    v2x, v2y = X[:, 2] - X[:, 0], Y[:, 2] - Y[:, 0]

    # Compute twice the area of each triangle
    area2 = np.abs(v1x * v2y - v1y * v2x)  # Shape: (N_E,)
    area = 0.5 * area2

    # Compute gradients of basis functions
    dphi1 = np.stack([(Y[:, 1] - Y[:, 2]), (X[:, 2] - X[:, 1])], axis=1) / area2[:, None]
    dphi2 = np.stack([(Y[:, 2] - Y[:, 0]), (X[:, 0] - X[:, 2])], axis=1) / area2[:, None]
    dphi3 = np.stack([(Y[:, 0] - Y[:, 1]), (X[:, 1] - X[:, 0])], axis=1) / area2[:, None]

    # Compute the gradient of u^n in each triangle
    U_local = current_iterate[elements]  # Shape: (N_E, 3)
    grad_u = (U_local[:, 0, None] * dphi1 +
              U_local[:, 1, None] * dphi2 +
              U_local[:, 2, None] * dphi3)  # Shape: (N_E, 2)

    # Compute |∇u^n|^2
    grad_u_sq = np.sum(grad_u**2, axis=1)  # Shape: (N_E,)

    # Compute integral |T| * |∇u^n|^2
    element_integrals = area * grad_u_sq  # Shape: (N_E,)

    # Get cubature rule
    rule = get_rule(rule=cubature_rule)
    weights = rule.weights_and_integration_points.weights
    integration_points = rule.weights_and_integration_points.integration_points

    rhs_integrals = np.zeros(elements.shape[0])
    for weight, (eta, xi) in zip(weights, integration_points):
        transformed_points = z_0 + eta * (z_1 - z_0) + xi * (z_2 - z_0)
        f_values = f(transformed_points)
        phi = np.array([1.-eta-xi, eta, xi])
        U_on_elements = np.sum(U_local * phi, axis=1)
        rhs_integrals += weight * f_values * U_on_elements

    rhs_integrals *= area2

    return -rhs_integrals + 0.5 * element_integrals
