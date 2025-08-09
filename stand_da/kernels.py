from numba import cuda, float64

TPB = 32
@cuda.jit
def MatMulMat(A, B, C):
    """
    Perform matrix multiplication of C = A * B using CUDA shared memory.

    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float64(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

# @cuda.jit(device=True)
# def linear_inequality(u, v, itv): #u + vz < 0
#     if v == 0:
#         if (u <= 0):
#             return itv
#         else:
#             return None
#     if (v < 0):
#         if (-u/v > itv[0]):
#             itv[0] = -u/v
#     if (-u/v < itv[1]):
#         itv[1] = -u/v
#     return itv

@cuda.jit
def MatAddBias(A, B, C):
    # Use 1D grid for better memory coalescing
    idx = cuda.grid(1)
    if idx < C.size:
        row = idx // C.shape[1]
        col = idx % C.shape[1]
        C[row, col] = A[row, col] + B[col, 0]

@cuda.jit
def GetReluInterval(a, b, X, itv):
    x, y = cuda.grid(2)
    if (x < a.shape[0] and y < a.shape[1]):
        if X[x, y] < 0:
            if (b[x, y] != 0):
                if b[x, y] < 0: 
                    cuda.atomic.max(itv, 0, -a[x, y] / b[x, y])
                else:
                    cuda.atomic.min(itv, 1, -a[x, y] / b[x, y])
                    
            X[x, y] = 0
            a[x, y] = 0
            b[x, y] = 0
        else:
            if (-b[x, y] != 0):
                if -b[x, y] < 0:
                    cuda.atomic.max(itv, 0, -a[x, y] / b[x, y])
                else:
                    cuda.atomic.min(itv, 1, -a[x, y] / b[x, y])

@cuda.jit 
def MatAddMat(A, B, C):
    idx = cuda.grid(1)
    if idx < C.size:
        row = idx // C.shape[1]
        col = idx % C.shape[1]
        C[row, col] = A[row, col] + B[row, col]

@cuda.jit
def MatMulNum(A, num, C):
    idx = cuda.grid(1)
    if idx < C.size:
        row = idx // C.shape[1]
        col = idx % C.shape[1]
        C[row, col] = A[row, col] * num