"""
Poisson Image Editing
William Emmanuel
wemmanuel3@gatech.edu
CS 6745 Final Project Fall 2017
"""

import numpy as np
from scipy.sparse import lil_matrix as lil_matrix
from scipy.sparse import linalg as splinalg
from scipy.sparse import csr_matrix

# Helper enum
OMEGA = 0
DEL_OMEGA = 1
OUTSIDE = 2

# Determine if a given index is inside omega, on the boundary (del omega),
# or outside the omega region
def point_location(index, mask):
    if in_omega(index,mask) == False:
        return OUTSIDE
    if edge(index,mask) == True:
        return DEL_OMEGA
    return OMEGA

# Determine if a given index is either outside or inside omega
def in_omega(index, mask):
    return mask[index] == 1

# Deterimine if a given index is on del omega (boundary)
def edge(index, mask):
    if in_omega(index,mask) == False: return False
    for pt in get_surrounding(index):
        # If the point is inside omega, and a surrounding point is not,
        # then we must be on an edge
        if in_omega(pt,mask) == False: return True
    return False

# Apply the Laplacian operator at a given index
def lapl_at_index(source, index):
    i,j = index
    val = (4 * source[i,j])    \
           - (1 * source[i+1, j]) \
           - (1 * source[i-1, j]) \
           - (1 * source[i, j+1]) \
           - (1 * source[i, j-1])
    return val

# Find the indicies of omega, or where the mask is 1
def mask_indicies(mask):
    nonzero = np.nonzero(mask)
    return zip(nonzero[0], nonzero[1])

# Get indicies above, below, to the left and right
def get_surrounding(index):
    i,j = index
    return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]

# Create the A sparse matrix
def poisson_sparse_matrix(points, point_to_idx):
    N = len(points)
    row, col, data = [], [], []

    for idx, point in enumerate(points):
        row.append(idx)
        col.append(idx)
        data.append(4)

        for nb in get_surrounding(point):
            if nb in point_to_idx:
                j = point_to_idx[nb]
                row.append(idx)
                col.append(j)
                data.append(-1)

    A = csr_matrix((data, (row, col)), shape=(N, N))
    return A


def process(source, target, mask):
    indicies = list(mask_indicies(mask))
    N = len(indicies)

    point_to_idx = {pt: idx for idx, pt in enumerate(indicies)}
    A = poisson_sparse_matrix(indicies, point_to_idx)

    from scipy.ndimage import laplace
    source_lap = laplace(source, mode='nearest')

    b = np.zeros(N)
    for i, index in enumerate(indicies):
        b[i] = source_lap[index]
        if point_location(index, mask) == DEL_OMEGA:
            # Faster: sum boundary contributions with list comprehension
            b[i] += sum(target[pt] for pt in get_surrounding(index) if not in_omega(pt, mask))

    # Solve using conjugate gradient
    x, info = splinalg.cg(A, b, tol=1e-5, maxiter=1500)
    if info != 0:
        print(f"CG did not converge, falling back to spsolve (info: {info})")
        x = splinalg.spsolve(A, b)

    composite = np.copy(target).astype(np.float32)
    indices_i, indices_j = zip(*indicies)
    composite[indices_i, indices_j] = np.clip(x, 0, 255)

    return composite


# Naive blend, puts the source region directly on the target.
# Useful for testing
def preview(source, target, mask):
    return (target * (1.0 - mask)) + (source * (mask))

def poisson_blend(source, target, mask):
    # source_img = cv2.imread(source_names[0], cv2.IMREAD_COLOR)
    # target_img = cv2.imread(target_names[0], cv2.IMREAD_COLOR)
    # mask_img = cv2.imread(mask_names[0], cv2.IMREAD_GRAYSCALE)

    # Make mask binary
    mask[mask != 0] = 1
    channels = source.shape[-1]
    # Call the poisson method on each individual channel
    result_stack = [process(source[:,:,i], target[:,:,i], mask) for i in range(channels)]
    # Merge the channels back into one image
    result = np.stack(result_stack, axis=-1)
    return result