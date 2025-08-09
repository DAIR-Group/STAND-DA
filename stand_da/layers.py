from .operations import *
from numba import cuda

TPB = 16
def Relu(a, b, x, itv, streams):
    return GetReluInterval(a, b, x, itv, streams[0])

def LinearWeight(a, b, x, weight, streams):
    return MatMulMat(a, weight, streams[0]), MatMulMat(b, weight, streams[1]), MatMulMat(x, weight, streams[2])

def LinearBias(a, b, x, bias, streams):
    return MatAddBias(a, bias, streams[0]), b, MatAddBias(x, bias, streams[1])