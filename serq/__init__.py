import torch
from . import nn
import serq._CUDA

__all__ = [
    "matmul_lowrank",
    "matmul",
    "simple_quantize_mxfp4"
]

def flatten_last_dim_and_return_shape(x: torch.Tensor):
    shape_excl_last = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    return x, shape_excl_last

def matmul_lowrank(A, B, C, SFA, SFB):
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    C, C_shape_excl_last = flatten_last_dim_and_return_shape(C)
    return serq._CUDA.matmul_lowrank(A, B, C, SFA, SFB).view(*C_shape_excl_last, -1)

def matmul(A, B, SFA, SFB):
    A, A_shape_excl_last = flatten_last_dim_and_return_shape(A)
    B, B_shape_excl_last = flatten_last_dim_and_return_shape(B)
    return serq._CUDA.matmul(A, B, SFA, SFB).view(*A_shape_excl_last, -1)

def simple_quantize_mxfp4(x):
    x, x_shape_excl_last = flatten_last_dim_and_return_shape(x)
    quantized_data, scale = serq._CUDA.simple_quantize_mxfp4(x)
    quantized_data = quantized_data.view(*x_shape_excl_last, -1)
    return quantized_data, scale

class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype