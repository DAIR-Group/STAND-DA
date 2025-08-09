from . import kernels as kns
from numba import cuda
import math
import numpy as np

TPB = 32

def convert_network_to_numpy(model):
    layers = []
    list_layers = []
    ptr = 0
    for name, param in model.named_children():
        temp = dict(param._modules)
        
        for layer_name in temp.values():
            if ('Linear' in str(layer_name)):
                layers.append('Linear')
            elif ('ReLU' in str(layer_name)):
                layers.append('ReLU')
    for name, param in model.named_parameters():
        if (layers[ptr] == 'Linear'):
            if ('weight' in name):
                weight = np.asarray(param.data.cpu())
                list_layers.append((layers[ptr] + ' ' + 'Weight', cuda.to_device(weight.T)))
            elif ('bias' in name):
                bias = np.asarray(param.data.cpu()).reshape(-1, 1)
                list_layers.append((layers[ptr] + ' ' + 'Bias', cuda.to_device(bias)))
                ptr += 1

        if (ptr < len(layers) and layers[ptr] == 'ReLU'):
            list_layers.append((layers[ptr], None))
            ptr += 1
    return list_layers

def MatMulMat(A, B, stream):
    C = cuda.device_array((A.shape[0], B.shape[1]))
    threadsperblock = (TPB, TPB)
    grid_y_max = max(A.shape[0], B.shape[0])
    grid_x_max = max(A.shape[1], B.shape[1])
    blockspergrid_x = math.ceil(grid_x_max / threadsperblock[0])
    blockspergrid_y = math.ceil(grid_y_max / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    kns.MatMulMat[blockspergrid, threadsperblock, stream](A, B, C)
    return C

def MatAddBias(A, B, stream):
    C = cuda.device_array((A.shape[0], A.shape[1]))
    # Use 1D grid to match the kernel
    threadsperblock = TPB * TPB
    blockspergrid = math.ceil(C.size / threadsperblock)
    kns.MatAddBias[blockspergrid, threadsperblock, stream](A, B, C)
    return C

def MatAddMat(A, B):
    C = cuda.device_array((A.shape[0], A.shape[1]))
    threadsperblock = TPB * TPB
    blockspergrid = math.ceil(C.size / threadsperblock)
    kns.MatAddMat_Optimized[blockspergrid, threadsperblock](A, B, C)
    return C

def MatMulNum(A, num):
    C = cuda.device_array((A.shape[0], A.shape[1]))
    threadsperblock = TPB * TPB
    blockspergrid = math.ceil(C.size / threadsperblock)
    kns.MatMulNum_Optimized[blockspergrid, threadsperblock](A, num, C)
    return C

def GetReluInterval(a, b, x, itv, stream):
    threadsperblock = (TPB, TPB)
    blockspergrid = (int(math.ceil(a.shape[0] / threadsperblock[0])), int(math.ceil(a.shape[1] / threadsperblock[1])))
    kns.GetReluInterval[blockspergrid, threadsperblock, stream](a, b, x, itv)
    return a, b, x, itv
