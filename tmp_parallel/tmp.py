import numpy as np
import time
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

array = np.array([0., 1.])
alpha = 4.5


def compute_edge(array: np.ndarray, alpha: float, edge: np.ndarray):
    time.sleep(0.001)

    edge_norm = np.linalg.norm(edge)
    max = np.max(array) * alpha

    return edge_norm, max


def perform_parallel(edges) -> tuple[np.ndarray, np.ndarray]:
    # Use all available CPU cores
    n_cores = cpu_count()
    print(f'performing EVA on {n_cores} cores...')
    with Pool(n_cores) as pool:
        results = pool.starmap(
            compute_edge, [(array, alpha, e) for e in edges])
        print(results)

    # Unpacking the two result arrays
    norms, maxes = zip(*results)
    return np.array(norms), np.array(maxes)


def perform_serial(edges) -> list[float]:
    norms, maxes = [], []
    for edge in edges:
        norm, max = compute_edge(array, alpha, edge)
        norms.append(norm)
        maxes.append(max)
    return np.array(norms), np.array(maxes)


if __name__ == "__main__":
    # Generate random edges (N x 2)
    N = 10000
    edges = np.random.rand(N, 2)

    start_time = time.time()
    results_parallel = perform_parallel(edges)
    end_time = time.time()
    time_parallel = end_time - start_time

    start_time = time.time()
    results_serial = perform_serial(edges)
    end_time = time.time()
    time_serial = end_time - start_time

    sames = [rl == rs for (rl, rs) in zip(results_parallel, results_serial)]
    print(np.all(sames))
    
    print(f"Time parallel: {time_parallel:.4f} seconds")
    print(f"Time serial: {time_serial:.4f} seconds")
