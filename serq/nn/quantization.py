import serq
import torch

class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        quantized_x, scales_x = serq.simple_quantize_mxfp4(x)
        packed_tensor = serq.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor