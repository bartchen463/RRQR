import numpy as np

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


def RRQR(A, delta):
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

A = np.random.uniform(low = -10, high = 10, size = (20,20))
for i in range(10):
    inds = np.random.choice(range(10), 4, replace = False)
    x = np.zeros(20)
    for ind in inds:
        u = np.random.uniform(size = 20)
        x += np.random.randint(-5,5)*A[:,ind] + u/10
    A[:,i+10] = x


P, Q, R, k, epsilon = RRQR(A, 0)

P1, Q1, R1, k1, epsilon1 = RRQR(A, 0.5)

np.linalg.norm(A@P1 - Q1@R1)


AP = A@P
AP1 = Q@R

n = 79
m = 60
A = np.array([[np.random.randint(-6,19) for i in range(n)] for j in range(m)]).astype(float)
P, Q, R = RRQR(A, 0.05)



