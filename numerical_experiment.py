import numpy as np
import matplotlib.pyplot as plt
from TestQR import *

def generate_rank_deficient_matrix(m, n, rank):
    U = np.random.rand(m, rank)  # Random matrix with m rows and "rank" columns
    V = np.random.rand(rank, n)  # Random matrix with "rank" rows and n columns
    return U @ V  # Resulting matrix has at most "rank" non-zero singular values

# Generate a nearly rank-deficient matrix
def generate_nearly_rank_deficient_matrix(m, n, deficiency, epsilon=1e-3):
    rank = max(min(m, n) - deficiency, 1)  # Ensure rank is at least 1
    low_rank_matrix = generate_rank_deficient_matrix(m, n, rank)
    perturbation = epsilon * np.random.rand(m, n)  # Add small random noise
    return low_rank_matrix + perturbation

def test_rrqr_performance(matrix, epsilon):
    frob_norm = np.linalg.norm(matrix, ord="fro")
    delta = epsilon * frob_norm 
    P, Q, R = RRQR(matrix, delta)
    rank = np.sum(np.abs(np.diag(R)) > delta)  # Numerical rank estimate

    # Truncated matrices for low-rank approximation
    Q_k = Q[:, :rank]
    R_k = R[:rank, :]
    approx_matrix = Q_k @ R_k @ P.T

    # Approx error
    frob_norm_diff = np.linalg.norm(matrix - approx_matrix, ord="fro")
    two_norm_diff = np.linalg.norm(matrix - approx_matrix, ord=2)

    # Error bound from the truncated part of R
    if rank < R.shape[0] and rank < R.shape[1]:
        R22 = R[rank:, rank:]
        error_bound = np.linalg.norm(R22, ord=2) 
    else:
        error_bound = 0 

    return {
        "rank": rank,
        "frob_norm_diff": frob_norm_diff,
        "two_norm_diff": two_norm_diff,
        "error_bound": error_bound,
    }

def collect_rrqr_results():
    size = 100  
    deficiencies = range(1, 11)  # Rank deficiencies 
    epsilons = np.logspace(-6, -2, 10)  
    results = []

    for deficiency in deficiencies:
        matrix = generate_nearly_rank_deficient_matrix(size, size, deficiency)
        frob_norm = np.linalg.norm(matrix, ord="fro")
        for epsilon in epsilons:
            metrics = test_rrqr_performance(matrix, epsilon)
            results.append((deficiency, epsilon, metrics))

    return results

# Plotting results
def plot_rrqr_analysis(results):
    deficiencies = sorted(set(result[0] for result in results)) 

    # Delta vs. Rank
    plt.figure(figsize=(10, 6))
    for deficiency in deficiencies:
        deficiency_results = [result for result in results if result[0] == deficiency]
        epsilons = [result[1] for result in deficiency_results]
        ranks = [result[2]["rank"] for result in deficiency_results]
        plt.semilogx(epsilons, ranks, marker="o", label=f"Near-Rank Gap  {deficiency}")
    plt.title("Tolerance vs. Rank for 100x100 Matrix")
    plt.xlabel("Tolerance")
    plt.ylabel("Rank")
    plt.legend()
    plt.grid()
    plt.show()

    # Delta vs. Approximation Error (2-Norm)
    plt.figure(figsize=(10, 6))
    for deficiency in deficiencies:
        deficiency_results = [result for result in results if result[0] == deficiency]
        epsilons = [result[1] for result in deficiency_results]
        errors = [result[2]["two_norm_diff"] for result in deficiency_results]
        plt.semilogx(epsilons, errors, marker="o", label=f"Near-Rank Gap {deficiency}")
    plt.title("Tolerance vs. 2-Norm Difference for 100x100 Matrix")
    plt.xlabel("Tolerance")
    plt.ylabel("2-Norm Difference")
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

    # Rank vs. 2-Norm Difference
    plt.figure(figsize=(10, 6))
    for deficiency in deficiencies:
        deficiency_results = [result for result in results if result[0] == deficiency]
        ranks = [result[2]["rank"] for result in deficiency_results]
        norm_diffs = [result[2]["two_norm_diff"] for result in deficiency_results]
        plt.plot(ranks, norm_diffs, label=f"Near-Rank Gap {deficiency}", alpha=0.7)
    plt.title("Rank vs. 2-Norm Difference for 100x100 Matrix")
    plt.xlabel("Rank")
    plt.ylabel("2-Norm Difference")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    results = collect_rrqr_results()
    plot_rrqr_analysis(results)
