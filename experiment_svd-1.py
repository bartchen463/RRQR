from re import T
from tkinter import S
import numpy as np
import matplotlib.pyplot as plt
from TestQR import *

# Goal: Show that if singular values decay rapidly, then RRQR gives near-optimal results 
# TO DO: test the error of truncated SVD against that of RRQR for different matrices w unique singular value distrubitions
#        graph results to display the results

def create_matrix(n,m, singular_values):
    U = np.random.randn(m,m)
    V = np.random.randn(n,n)
    
    # want random, but orthogonal matrices for the SVD structure
    U, _ = np.linalg.qr(U)
    V, _ = np.linalg.qr(V)
    
    l = min(m,n)
    t = l/2
    j = l - l/2
    
    #create decaying singular values
    #singular_values = np.array([ 1/s**2 for s in np.arange(1,l+1)])
    
    if n > m:
        #horizontal matrix with n<m
        return U[:, :l] @ np.diag(singular_values) @ V.T[:l,:]
    elif n <= m:
        #vertical matrix with m <n
        return U[:, :l] @ np.diag(singular_values) @ V[:l,:].T



def compute_truncated_SVD(matrix, rank):
    #epsilon = 0 
    U, S, Vt = np.linalg.svd(matrix)
    #rank = np.sum(np.abs(np.diag(S)) > epsilon)  # Numerical rank estimate

    # Truncated matrices for low-rank approximation
    U_trunc = U[:,:rank]
    S_full_trunc = np.diag(S)[:rank, :rank]
    Vt_trunc = Vt[:rank, :]
    
    matrix_new = U_trunc @ S_full_trunc @ Vt_trunc
    return matrix_new





def PartialQR(A, k):
    R22 = A.copy()[k:,k:] ## Extract un-triagularized block R22
    m, n = A.shape
    u = R22[:,0]
    d = u.shape[0]
    unorm = np.linalg.norm(u)
    alpha = np.sign(u[0]) * unorm
    e1 = np.zeros(d)
    e1[0] = 1
    v = u - alpha * e1
    nv = v / np.linalg.norm(v)
    H = np.eye(d) - 2 * np.outer(nv, nv) ## Householder Matrix H
    Qfull = np.eye(m) ## Increase Dimensions H
    Qfull[k:, k:] = H
    R = Qfull@A ## Update R
    return Qfull, R


def RRQR2(A, delta, r):
    m, n = A.shape
    maxrank = np.min(A.shape) ## A can have at most rank min(n,m)
    Qf = np.eye(m)
    P = np.eye(n) # Initialize Permutation P = I
    k = 0
    R = A.copy()
    norms = np.linalg.norm(R, axis=0)
    epsilon = delta * np.sqrt(2) / (n+1)

    while np.max(norms) > 1e-10 and k != r - 1:
        maxcol = np.where(norms == np.max(norms))[0][0] + k
        x, y = P.copy()[:,k], R.copy()[:,k]
        P[:,k], R[:,k] = P[:,maxcol], R[:,maxcol] ## Swap columns based on norm of cols in lower block Ck
        P[:,maxcol], R[:,maxcol] = x, y
        Qk, R = PartialQR(R, k)
        Qf = Qf @ Qk ## Update Q
        if k + 1 != maxrank:
            Ck = R[k + 1:, k + 1:]
            norms = np.linalg.norm(Ck, axis=0)
            k += 1
            epsilon = np.sqrt(2) * delta / (n - k + 1)
    return P, Qf, R, k, epsilon

def compute_RRQR(matrix, rank):
    P, Q, R, k, epsilon = RRQR2(matrix, 0, rank)
    
    #rank = np.sum(np.abs(np.diag(R)) > delta)  # Numerical rank estimate
    # Truncated matrices for low-rank approximation
    matrix_k = Q @ R @ P.T
    return matrix_k

def testing_SVD(n,m):   
    
    r = 20
    
    norm_errors1 = []
    mat_1_k = create_matrix(n,m, np.array([ 1/s for s in np.arange(1,n+1)]))
     
    for k in range(r):   
        mat_SVD1 = compute_truncated_SVD(mat_1_k, k)
        norm_errors1.append(np.linalg.norm(mat_1_k - mat_SVD1, ord ='fro')) 
        
    norm_errors2 = []
    mat_1_k2 = create_matrix(n,m, np.array([ 1/s**2 for s in np.arange(1,n+1)]))
     
    for k in range(r):   
        mat_SVD2 = compute_truncated_SVD(mat_1_k2, k)
        norm_errors2.append(np.linalg.norm(mat_1_k2 - mat_SVD2, ord ='fro')) 
    
    norm_errors3 = []
    rank_e = np.array([ np.exp(-s*0.6)  for s in np.arange(1,n+1)])
    mat_e = create_matrix(n,m, rank_e)
    
    for k in range(r):   
        mat_SVD3 = compute_truncated_SVD(mat_e, k)
        norm_errors3.append(np.linalg.norm(mat_e - mat_SVD3, ord ='fro')) 

    norm_errors4 = []
    rank_rrqr = np.array([ 1/s**2  for s in np.arange(1,n+1)])
    mat_rrqr = create_matrix(n,m, rank_rrqr)
    
    for k in range(r):   
        mat_R = compute_RRQR(mat_rrqr, k)
        norm_errors3.append(np.linalg.norm(mat_rrqr - mat_R, ord ='fro')) 
        
        
    rank_vals = range(r)
    
   # print("rank_vals length", len(rank_vals))
    #print("norm_errors ", len(norm_errors1))
    
    
    plt.plot(rank_vals, norm_errors1, label = "= 1/k", color = 'm')
    plt.plot(rank_vals, norm_errors2, label = "=1/k^2", color = 'b')
    plt.plot(rank_vals, norm_errors3, label = "=1/e^-nk", color = 'g')
    plt.plot(rank_vals, norm_errors4, label = "rrrqr", color = 'y')
    
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Frobenius-Norm of Approximation Error')
    plt.title("Truncated SVD for Different SV Decaying Distributions")
    plt.legend()
    plt.show() 
    
    return 0


testing_SVD(100,100)
    
    
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

#graphing_comparison(20,30,10)
    





#OTHER IDEAS FOR CREATING SINGULAR VALUE DISTRIBUTIONS
    
    #create increasingly decaying values 
    #small_sv = np.random.uniform(0.0001, 0.01, int(t))
    #large_sv = np.random.uniform(1, 10, int(j))
    
    #singular_values = np.sort(np.concatenate((small_sv, large_sv)))