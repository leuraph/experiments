from scipy.sparse import csc_array, csr_array
import numpy as np
from my_timer import Timer

# Run with python -m tests.manual.<this_files_name>

# Note
# ----
#
# This experiment was supposed to help decide on what combination of
# CSC, CSR arrays to use when computing expressions like v.T * A * v =: a(v,v)
#
# Result
# ------
#
# It is quite clear, that we profit the most when using
# v: np.ndarray
# sA_csr: scipy.sparse.csr_array
# and do `v.dot(sA_csr.dot(v))`


def measure_function_call(func, *args, **kwargs) -> float:
    my_timer = Timer()
    my_timer.start()
    func(*args, **kwargs)
    my_timer.stop()
    return my_timer.read()


def measure_cpu(func):
    def wrapper_measure_cpu(*args, **kwargs):
        return measure_function_call(func, *args, **kwargs)
    return wrapper_measure_cpu


def measure_cpu_averaged(func):
    def wrapper_measure_cpu_averaged(*args, **kwargs):
        cpu_times = []
        for _ in range(10000):
            cpu_times.append(measure_function_call(func, *args, **kwargs))
        return np.mean(cpu_times), np.std(cpu_times)
    return wrapper_measure_cpu_averaged


@measure_cpu_averaged
def measure_left_multiplication(linear_operator, vector):
    return linear_operator.dot(vector)


@measure_cpu_averaged
def measure_right_multiplication(linear_operator, vector):
    return vector.dot(linear_operator)


@measure_cpu_averaged
def measure_bilinear_product(linear_operator, vector_left, vector_right):
    return vector_left.dot(linear_operator.dot(vector_right))


@measure_cpu_averaged
def measure_inner_product(vector_left, vector_right):
    return vector_left.dot(vector_right)


@measure_cpu_averaged
def measure_transposition(array):
    array.transpose()


def get_nice_string(dt_ns_mean: float, dt_ns_std: float):
    return f"({dt_ns_mean*1e-9:.3} ± {dt_ns_std*1e-9:.3}) s"


def main() -> None:
    dim = 3000
    my_timer = Timer()

    # Matrices
    print("building matrices/arrays...")
    my_timer.start()
    A = np.identity(dim)
    v = np.zeros(dim)
    v[[0, 1, 2]] = [1, 1, 1]
    v_csr = csc_array(v)
    v_csc = csc_array(v)
    A_csc = csc_array(A)
    A_csr = csr_array(A)
    my_timer.stop()
    print(
        "done building matrices/arrays in dt = "
        f"{my_timer.read()*1e-9:.3} s.")

    print("performing experiments...")

    dt_full, dt_full_std = measure_bilinear_product(A, v, v)
    dt_sparse_csc, dt_sparse_csc_std = measure_bilinear_product(A_csc, v, v)
    dt_sparse_csr, dt_sparse_csr_std = measure_bilinear_product(A_csr, v, v)

    dt_dot_csr, dt_dot_csr_std = measure_inner_product(
        v_csc, v_csc.transpose())
    dt_dot_csc, dt_dot_csc_std = measure_inner_product(
        v_csc, v_csc.transpose())
    dt_dot_np, dt_dot_np_std = measure_inner_product(v, v)

    dt_transposition_matrix_csr, dt_transposition_matrix_csr_std = \
        measure_transposition(A_csr)
    dt_transposition_matrix_csc, dt_transposition_matrix_csc_std = \
        measure_transposition(A_csc)
    dt_transposition_matrix_np, dt_transposition_matrix_np_std = \
        measure_transposition(A)

    dt_transposition_vector_csr, dt_transposition_vector_csr_std = \
        measure_transposition(v_csr)
    dt_transposition_vector_csc, dt_transposition_vector_csc_std = \
        measure_transposition(v_csc)
    dt_transposition_vector_np, dt_transposition_vector_np_std = \
        measure_transposition(v)

    print("done performing experiments.")
    print("----------------------------")
    print("")
    print("v * A * v  (np): " + get_nice_string(dt_full, dt_full_std))
    print("v * A * v (csc): " + get_nice_string(
        dt_sparse_csc, dt_sparse_csc_std))
    print("v * A * v (csr): " + get_nice_string(
        dt_sparse_csr, dt_sparse_csr_std))
    print("<v,v>     (csr): " + get_nice_string(dt_dot_csr, dt_dot_csr_std))
    print("<v,v>     (csc): " + get_nice_string(dt_dot_csc, dt_dot_csc_std))
    print("<v,v>      (np): " + get_nice_string(dt_dot_np, dt_dot_np_std))

    print("A.T       (csr): " + get_nice_string(
        dt_transposition_matrix_csr, dt_transposition_matrix_csr_std))
    print("A.T       (csc): " + get_nice_string(
        dt_transposition_matrix_csc, dt_transposition_matrix_csc_std))
    print("A.T        (np): " + get_nice_string(
        dt_transposition_matrix_np, dt_transposition_matrix_np_std))

    print("v.T       (csr): " + get_nice_string(
        dt_transposition_vector_csr, dt_transposition_vector_csr_std))
    print("v.T       (csc): " + get_nice_string(
        dt_transposition_vector_csc, dt_transposition_vector_csc_std))
    print("v.T        (np): " + get_nice_string(
        dt_transposition_vector_np, dt_transposition_vector_np_std))


if __name__ == "__main__":
    main()