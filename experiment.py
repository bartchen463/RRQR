import numpy as np
import matplotlib.pyplot as plt
from TestQR import *

def generate_test_matrices(N):
    return np.random.rand(N, N)

# Apply RRQR and Test Metrics
def test_rrqr(matrix, delta):
    P, Q, R = RRQR(matrix, delta)
    rank = np.sum(np.abs(np.diag(R)) > delta)
    approx_matrix = Q @ R @ P.T
    norm_diff = np.linalg.norm(matrix - approx_matrix, ord=2)
    return rank, norm_diff

# Experiment and Plot Results
def evaluate_and_plot():
    matrices = {N : generate_test_matrices(N) for N in range(10, 101, 10)}
    deltas = np.logspace(-5, -1, 5)  # Thresholds for rank approximation
    results = {}

    for size, matrix in matrices.items():
        ranks = []
        norm_diffs = []
        for delta in deltas:
            rank, norm_diff = test_rrqr(matrix, delta)
            ranks.append(rank)
            norm_diffs.append(norm_diff)

        results[size] = {"ranks": ranks, "norm_diffs": norm_diffs}

        # Plot Results
        plt.figure()
        plt.plot(ranks, norm_diffs, marker="o", label=size)
        plt.title(f"Performance on {size} matrix")
        plt.xlabel("Rank")
        plt.ylabel("2-Norm Difference")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    evaluate_and_plot()
