import numpy as np


def direct_error_estimator(
        u_new: np.ndarray,
        u_old: np.ndarray,
        dt: float,
        beta: float,
        eta_m: float,
        norm) -> float:
    return (
        (1. + (beta)/(dt * eta_m))
        *
        norm(u=u_new - u_old)
    )


def relaxation_error_estimator(
        alpha: float,
        eta_m: float,
        beta: float,
        dt: float,
        grad_energy_norm: float) -> float:
    return (
        (grad_energy_norm/eta_m)
        *
        np.exp(-(alpha*eta_m)/(beta**2) * dt)
    )


def euler_error_estimator_2(
        u_old,
        u_new,
        alpha: float,
        beta: float,
        eta_m: float,
        L: float,
        dt: float,
        norm
) -> float:
    return (
        (L/eta_m)
        *np.sqrt(beta/alpha)
        *norm(u_old - u_new)
        *np.sqrt(1. - np.exp(- eta_m/beta * dt))
    )


def euler_error_estimator_3(
        u_old: np.ndarray,
        u_new: np.ndarray,
        alpha: float,
        beta: float,
        dt: float,
        norm,
        E
) -> float:
    """
    Returns an error estimate for the euler error
    for one (restarted) step of size dt, i.e.,

    parameters
    ----------
    u_old: np.ndarray
        u^n_N
    u_new: np.ndarray
        u^{n+1}_N
    alpha: float
        coercivity constant of A
    beta: float
        boundedness constant of A
    dt: float
        step-size
    norm
        Hilbert space norm, i.e. norm(u) := \|u\|_X
    callable energy functional
        E(u: np.ndarray)
    """
    err_squared = (
        -(dt**2/alpha)*(
            (E(u_old) - E(u_new))/(dt)
            -
            beta * norm((u_new - u_old)/dt)**2
        )
    )
    return np.sqrt(err_squared)


def euler_error_estimator_4() -> float:
    pass



def get_restarted_estimator(
        grad_energy_norm: np.ndarray,
        eta_m: float,
        alpha: float,
        beta: float,
        L: float,
        dt: float
) -> float:
    return grad_energy_norm * (
        (
            (1./eta_m)
            * np.exp(-(alpha*eta_m)/(beta**2) * dt)
        )
        +
        (
            np.sqrt(beta/alpha)
            * (L / eta_m)
            * dt
            * np.sqrt(1. - np.exp(- eta_m/beta * dt))
        )
    )