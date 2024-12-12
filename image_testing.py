import numpy as np
from scipy import linalg as la
from TestQR import *
from PIL import Image
import matplotlib.pyplot as plt
import os


def image_to_grids(image, grid_size):
    h, w = image.shape[:2]
    pad_h = (grid_size - h % grid_size) % grid_size
    pad_w = (grid_size - w % grid_size) % grid_size

    # Add padding to the image
    if image.ndim == 2:  # Grayscale
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    else:  # RGB
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Split into grids and flatten each grid into 1D
    grids = []
    for i in range(0, padded_image.shape[0], grid_size):
        for j in range(0, padded_image.shape[1], grid_size):
            grid = padded_image[i:i+grid_size, j:j+grid_size]
            grids.append(grid.flatten())  # Flatten into 1D for both grayscale and RGB

    return np.array(grids).T  # Return as a 2D matrix with grids as columns


def grids_to_image(grids, grid_size, image_shape):
    h, w = image_shape[:2]
    pad_h = h + (grid_size - h % grid_size) % grid_size
    pad_w = w + (grid_size - w % grid_size) % grid_size
    c = image_shape[2] if len(image_shape) == 3 else 1  # Determine number of channels (1 for grayscale, 3 for RGB)

    # Initialize the reconstructed padded image
    if c == 1:  # Grayscale
        recon_image = np.zeros((pad_h, pad_w))
    else:  # RGB
        recon_image = np.zeros((pad_h, pad_w, c))

    # Rebuild the image grid by grid
    idx = 0
    for i in range(0, pad_h, grid_size):
        for j in range(0, pad_w, grid_size):
            if c == 1:  # Grayscale
                recon_image[i:i+grid_size, j:j+grid_size] = grids[:, idx].reshape((grid_size, grid_size))
            else:  # RGB
                recon_image[i:i+grid_size, j:j+grid_size, :] = grids[:, idx].reshape((grid_size, grid_size, c))
            idx += 1

    # Crop to the original size
    return recon_image[:h, :w] if c == 1 else recon_image[:h, :w, :]

def truncated_svd(image_matrix, rank_s):
    """
    Compute the truncated SVD of the given image matrix.
    Parameters:
    - image_matrix: 2D numpy array (grayscale image or channel of RGB image)
    - rank_s: Desired rank for the approximation
    Returns:
    - Approximation of the image matrix with rank rank_s
    """
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    U_s = U[:, :rank_s]
    S_s = np.diag(S[:rank_s])
    Vt_s = Vt[:rank_s, :]
    return U_s @ S_s @ Vt_s


# Save the original image to file and get its size
original_path = "C:/Users/oweno/OneDrive/Documents/College/Undergrad Physics/Year 3/1st Semester/APPM4600/Final Project/RRQR/images/original_grayscale.png"
original_image = Image.open(original_path)
original_image_arr = np.array(original_image) / 255.  # Normalize to [0, 1]
original_size = os.path.getsize(original_path)

grid_sizes = [8*i for i in range(1, 17)] 
delta_base = 1e3

compression_ratios_rrqr = []
compression_ratios_svd = []
relative_errors_rrqr = []
relative_errors_svd = []
ranks_rrqr = []

# Process each grid size
for grid_size in grid_sizes:
    image_gridded = image_to_grids(original_image_arr, grid_size)
    col_norms = np.linalg.norm(image_gridded, axis=0)
    mean_norms = np.mean(col_norms)
    print(f'mean norms = {mean_norms}, image gridded shape = {image_gridded.shape}')
    delta = delta_base 

    # Apply RRQR
    P, Qf, R, k, epsilon = RRQR(image_gridded, delta)

    # Reconstruct the image
    Q_k = Qf[:, :k]
    R_k = R[:k, :]
    approx_gridded = Q_k @ R_k @ P.T
    approx_im_rrqr = grids_to_image(approx_gridded, grid_size, original_image_arr.shape)

    # Apply SVD
    if original_image_arr.ndim == 2:  # Grayscale
        rank_k = k
        approx_im_svd = truncated_svd(original_image_arr, rank_k)
    else:  # RGB
        rank_k = k
        approx_im_svd = np.stack([truncated_svd(original_image_arr[:, :, c], rank_k) for c in range(3)], axis=2)

    # Debugging: Check RRQR results
    
    rows = grid_size**2 * (1 if original_image_arr.ndim == 2 else 3)  # Grayscale or RGB
    columns = image_gridded.shape[1]  # Number of grids
    full_rank = min(rows, columns)
    print(f"Grid Size: {grid_size}, Full Rank: {full_rank}, Numerical Rank: {k}")

    # Save the reconstructed RRQR image to file and get its size
    compressed_path = f"C:/Users/oweno/OneDrive/Documents/College/Undergrad Physics/Year 3/1st Semester/APPM4600/Final Project/RRQR/images/rrqr_image_grid{grid_size}_rank{rank_k}.png"
    approx_image_arr = (approx_im_rrqr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(approx_image_arr).save(compressed_path)

    # Save the SVD image to file and get its size
    svd_path = f"C:/Users/oweno/OneDrive/Documents/College/Undergrad Physics/Year 3/1st Semester/APPM4600/Final Project/RRQR/images/svd_image_grid{grid_size}_rank{rank_k}.png"
    approx_image_svd = (approx_im_svd * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(approx_image_svd).save(svd_path)

    # Calculate metrics for RRQR
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio_rrqr = original_size / compressed_size
    rel_error_rrqr = np.linalg.norm(original_image_arr - approx_im_rrqr) / np.linalg.norm(original_image_arr)
    print(f"Grid Size: {grid_size}, Numerical Rank: {k}, RRQR Compression Ratio: {compression_ratio_rrqr:.2f}")
    print(f"Grid Size: {grid_size}, Approximation Error: {rel_error_rrqr:.4f}")


    # Calculate metrics for SVD
    svd_compressed_size = os.path.getsize(svd_path)
    compression_ratio_svd = original_size / svd_compressed_size
    rel_error_svd = np.linalg.norm(original_image_arr - approx_im_svd) / np.linalg.norm(original_image_arr)
    print(f"Rel SVD Error: {rel_error_svd:.4f}, SVD Compression Ratio: {compression_ratio_svd:.2f}")

    # Store results
    relative_errors_rrqr.append(rel_error_rrqr)
    relative_errors_svd.append(rel_error_svd)
    compression_ratios_rrqr.append(compression_ratio_rrqr)
    compression_ratios_svd.append(compression_ratio_svd)
    ranks_rrqr.append(k)

# Plot Compression Ratio vs Grid Size
plt.figure()
plt.plot(grid_sizes, compression_ratios_rrqr, label="RRQR", marker='o')
plt.plot(grid_sizes, compression_ratios_svd, label="SVD", marker='s')
plt.xlabel("Grid Size")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs Grid Size")
plt.legend()
plt.grid()

# Plot Relative Error vs Grid Size
plt.figure()
plt.plot(grid_sizes, relative_errors_rrqr, label="RRQR", marker='o')
plt.plot(grid_sizes, relative_errors_svd, label="SVD", marker='s')
plt.xlabel("Grid Size")
plt.ylabel("Relative Error")
plt.title("Relative Error vs Grid Size")
plt.legend()
plt.grid()

# Plot Compression Ratio vs Relative Error
plt.figure()
plt.scatter(relative_errors_rrqr, compression_ratios_rrqr, label="RRQR", marker='o')
plt.scatter(relative_errors_svd, compression_ratios_svd, label="SVD", marker='s')
plt.xlabel("Relative Error")
plt.ylabel("Compression Ratio")
plt.title("Compression Ratio vs Relative Error")
plt.legend()
plt.grid()

# Plot Compression Ratio vs Rank
plt.figure()
plt.scatter(ranks_rrqr, compression_ratios_rrqr, label='RRQR', marker='o')
plt.scatter(ranks_rrqr, compression_ratios_svd, label='SVD', marker='s')
plt.xlabel('Rank')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratio vs Rank')
plt.legend()
plt.grid()


plt.show()