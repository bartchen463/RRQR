import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import Image as im 
import time
from TestQR import *


filename = "big_tree.jpeg"

image = Image.open(filename)
mat = np.array(image) / 255

start_t = time.time()

#compute the factorization
U, S, Vt = np.linalg.svd(mat)

print("Dimensions of SVD: ", U.shape, S.shape, Vt.shape)


#S_keep = np.zeros(len(S))
#S_keep = np.array([])

epsilon = 1
i = 0
k = 0

#Only want to keep the largest singular values, larger than a given epsilon
for a in S:
    if a < epsilon:
        k = i 
    else:
        i +=1
        
S_keep = S[:k]           
S_full = np.diag(S_keep) # converts to diagonal matrix (the above function returns a 1D Array)

#k = len(S_keep) # test

print("s_keep:",S_keep) 

U_trunc = U[:,:k]
S_full_trunc = S_full[:k, :k]
Vt_trunc = Vt[:k, :]

print("Dimensions of SVD: ", U_trunc.shape, S_full_trunc.shape, Vt_trunc.shape)

A_k = U_trunc @ S_full_trunc @ Vt_trunc * 255.0
new_image = Image.fromarray(A_k).convert('L')


new_image.save('new_image.png')

end_t = time.time()
print("Length of Program (seconds):", end_t - start_t)

#Q, R = np.linalg.qr(test)




    
