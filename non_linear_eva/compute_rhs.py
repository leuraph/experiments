import sympy as sp
from sympy.utilities.lambdify import lambdastr


def print_rhs_problem_01():
    x, y = sp.symbols('x y')

    # Problem configuration
    # ---------------------
    # solution
    u = x*(x-1)*y*(y-1)
    # diffusion matrix
    a_11 = 1.
    a_12 = 0.
    a_21 = 0.
    a_22 = 1.
    A = sp.Matrix([[a_11, a_12], [a_21, a_22]])
    # reaction term
    phi = sp.exp(u)
    # ---------------------

    # gradients and divergences
    grad_u = sp.Matrix([sp.diff(u, x), sp.diff(u, y)])
    A_grad_u = A * grad_u
    div_A_grad_u = sp.diff(A_grad_u[0], x) + sp.diff(A_grad_u[1], y)

    # computing corresponding right-hand side f
    f = -div_A_grad_u + phi

    # printing f as string
    print('Problem 01')
    print('----------')
    print(lambdastr((x,y), f.simplify()))


def main() -> None:
    print_rhs_problem_01()


if __name__ == '__main__':
    main()
