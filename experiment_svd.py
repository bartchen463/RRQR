from re import T
import numpy as np
import matplotlib.pyplot as plt
from TestQR import *

# Goal: Show that if singular values decay rapidly, then RRQR gives near-optimal results 
# TO DO: test the error of truncated SVD against that of RRQR for different matrices w unique singular value distrubitions
#        graph results to display the results

def create_matrix(n,m):
    U = np.random.randn(m,m)
    V = np.random.randn(n,n)
    
    # want random, but orthogonal matrices for the SVD structure
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    
    l = min(m,n)
    t = l/2
    j = l - l/2
    
    #create decaying singular values
    singular_values = np.array([1/(s+1) for s in np.arange(l)])
    
    #create increasingly decaying values 
    #small_sv = np.random.uniform(0.0001, 0.01, int(t))
    #large_sv = np.random.uniform(1, 10, int(j))
    
    #singular_values = np.sort(np.concatenate((small_sv, large_sv)))
    
    #print("U:", U[:, :l])
    #print("sig:", np.diag(singular_values))
    #print("Vt:", V.T[:l,:])
    
    if n > m:
        #horizontal matrix with n<m
        return U[:, :l] @ np.diag(singular_values) @ V.T[:l,:]
    elif n <= m:
        #vertical matrix with m <n
        return U[:, :l] @ np.diag(singular_values) @ V[:l,:].T


#mat = create_matrix(5,4)
#print(mat)
#print("mat shape:", mat.shape)
#U, S, Vt = np.linalg.svd(mat)
#print(S)

def compute_truncated_SVD(matrix, epsilon):
    #epsilon = 0 
    U, S, Vt = np.linalg.svd(matrix)
    rank = np.sum(np.abs(np.diag(S)) > epsilon)  # Numerical rank estimate

    # Truncated matrices for low-rank approximation
    U_trunc = U[:,:rank]
    S_full_trunc = np.diag(S)[:rank, :rank]
    Vt_trunc = Vt[:rank, :]    
    return U_trunc @ S_full_trunc @ Vt_trunc



def compute_RRQR(matrix, rank):
    frob_norm = np.linalg.norm(matrix, ord="fro")
    epsilon = 0
    delta = epsilon * frob_norm 
    P, Q, R, k, epsilon = RRQR(matrix, delta)
    
    #rank = np.sum(np.abs(np.diag(R)) > delta)  # Numerical rank estimate
    
    # Truncated matrices for low-rank approximation
    matrix_k = Q[:, :rank] @ R[:rank, :] @ P.T
    return matrix_k, rank



def testing_SVD(matrix, epsilon):
    # 
    
    return 0
    
    
def comparison(n,m,top_rank):
    mat = create_matrix(n,m)
    print(mat)
    
    U, S, Vt = np.linalg.svd(mat)
    #Find singular values of matrix
    
    errors_SVD = []
    errors_RRQR = []

    #e = np.arange(-6, -2, 10) 
    
    for e in epsilon:
        
        mat_RRQR, rank = compute_RRQR(mat, e)
        err_RRQR = np.linalg.norm(mat - mat_RRQR, ord ='fro')
        errors_RRQR.append(err_RRQR)
        
        #computes truncated SVD with the same rank as RRQR
        mat_SVD = compute_truncated_SVD(mat, rank)
        err_SVD = np.linalg.norm(mat - mat_SVD, ord ='fro')
        errors_SVD.append(err_SVD)
        
        
    return S, errors_SVD, errors_RRQR, ranks

S, errors_SVD, errors_RRQR, ranks = comparison(20,30,10)

print("S: ", S)
print("errors_svd: ", errors_SVD)
print("errors_RRQR: ", errors_RRQR)
print("ranks: ", ranks)

    
    
    
def graphing_comparison(n,m, top_rank):
    
    singular_vals, errors_SVD, errors_RRQR, ranks = comparison(n,m,top_rank)
    
    plt.plot(ranks, errors_SVD, label = "Truncated SVD Error", color = 'm')
    plt.plot(ranks, errors_RRQR, label = "RRQR Error", color = 'b')
    
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frobenius-Norm of Approximation Error')
    plt.title("Comparison of Errors in Truncated SVD vs RRQR")
    plt.legend()
    plt.show()
    
    print("Singular values: ", singular_vals)

graphing_comparison(20,30,10)
    



