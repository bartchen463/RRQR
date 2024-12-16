import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scila

def PartialQR(A, k):
    R22 = A[k:,k:].copy() ## Extract un-triagularized block R22
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

    while np.max(norms) > epsilon and k != maxrank - 1:
        maxcol = np.where(norms == np.max(norms))[0][0] + k
        x, y = P[:,k].copy(), R[:,k].copy()
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

def RRQR3(A, delta, r):
    m, n = A.shape
    maxrank = np.min(A.shape) ## A can have at most rank min(n,m)
    Qf = np.eye(m)
    P = np.eye(n) # Initialize Permutation P = I
    k = 0
    R = A.copy()
    norms = np.linalg.norm(R, axis=0)
    epsilon = delta * np.sqrt(2) / (n+1)

    #while np.max(norms) > epsilon and k != maxrank - 1:
    while np.linalg.norm(A - Qf[:,:k]@R[:k,:]@P.T) >= delta and k != maxrank - 1:
        maxcol = np.where(norms == np.max(norms))[0][0] + k
        x, y = P[:,k].copy(), R[:,k].copy()
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

A = np.random.randn(4, 4)
A[:,2] = 0.5*A[:,0] - 0.4*A[:,1] + np.array([0.01, 0.01, 0.01, 0.01])

P, Q, R, k, eps = RRQR2(A, 0, 4)

A1 = Q@R



def to_latex_bmatrix(array):
    """Convert a NumPy array into a LaTeX bmatrix."""
    if len(array.shape) > 2:
        raise ValueError("bmatrix can only be created from 2D arrays.")
    lines = [" & ".join(map(str, row)) + r" \\" for row in array]
    print("\\begin{bmatrix}\n" + "\n".join(lines) + "\n\\end{bmatrix}")
    return

latex_bmatrix = to_latex_bmatrix(A.round(2))

A = np.random.randn(100, 100)

sv = np.sort(np.array([np.exp(-s*0.1) for s in range(1, 201)]))[::-1]
S  = np.diag(sv)
U = np.random.randn(200, 200)
V = np.random.randn(200, 200)
U , _ = np.linalg.qr(U)
V , _ = np.linalg.qr(V)
A = U@S@V.T
P, Q, R, _, eps = RRQR2(A, 0, 0)

epsilonvals = sv
truevals = np.zeros(200)
undervals = np.zeros(200)
overvals = np.zeros(200)
error_under = np.zeros(200)
error_over = np.zeros(200)
for i in range(200):
    epsilon = epsilonvals[i]
    truevals[i] = np.sum(sv>=epsilon)
    k_u = np.sum(np.diag(R)>=epsilon)
    undervals[i] = k_u
    error_under[i] = np.linalg.norm(A - Q[:,:k_u]@R[:k_u,:]@P.T, ord = 2)
    P1, Q1, R1, k, _ = RRQR2(A, epsilon, 0)
    error_over[i] = np.linalg.norm(A - Q1[:,:k]@R1[:k,:]@P1.T, ord = 2)
    overvals[i] = k + 1


plt.plot(epsilonvals, truevals, label=r'True $\epsilon$ Rank', color="g")
plt.plot(epsilonvals, undervals, label=r'Stop when $|r_{kk}|\leq\epsilon$', color="r")
plt.plot(epsilonvals, overvals, label=r'Stop when error $\leq\epsilon$ is guaranteed', color="b")
plt.ylabel(r'Approximate $\epsilon$ Rank')
plt.xlabel(r'$\epsilon$')
plt.title(r'Numerical Rank for Different Stopping Criteria')
plt.xscale("log")
plt.legend()
plt.show()

plt.plot(epsilonvals, error_under, label = r'Stop when $|r_{kk}|\leq\epsilon$', color="r")
plt.plot(epsilonvals, epsilonvals, label = r'Target Error $\epsilon$', color="g")
plt.plot(epsilonvals, error_over, label = r'Stop when error $\leq\epsilon$ is guaranteed', color="b")
plt.ylabel(r'2-Norm Error')
plt.xlabel(r'$\epsilon$')
plt.title(r'2-Norm Error for Different Stopping Criteria')
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

truevals = np.zeros(200)
aprxvals = np.zeros(200)
for i in range(200):
    P, Q, R, k, epsilon = RRQR3(A, epsilonvals[i], 0)
    aprxvals[i] = k + 1
    truevals[i] = np.sum(sv>=epsilonvals[i])
    P1, Q1, R1, k, _ = RRQR2(A, epsilonvals[i], 0)
    overvals[i] = k + 1

plt.plot(epsilonvals, truevals, label=r'True $\epsilon$ Rank', color="g")
plt.plot(epsilonvals, aprxvals, label=r'Brute force best $\epsilon$ rank', color="b")
plt.plot(epsilonvals, overvals, label=r'Stop when error $\leq\epsilon$ is guaranteed', color="r")
plt.xscale("log")
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'Approximate $\epsilon$ Rank')
plt.title(r'Approximate $\epsilon$ Rank for Different Stopping Criteria')
plt.legend()
plt.show()