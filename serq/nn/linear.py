import torch
import serq
from serq_quant.int_quant import SVDLinear

class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('SF_w', (torch.randint(1, 128, (self.out_features * self.in_features // 32,), dtype=torch.uint8, requires_grad=False)))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        pre_weight = torch.rand(self.out_features, self.in_features, dtype=torch.bfloat16).cuda()
        self.weight.data, self.SF_w.data = serq.simple_quantize_mxfp4(torch.abs(pre_weight))
        del pre_weight
        self.register_buffer('SF_low_rank_weight', (torch.randint(1, 128, (self.out_features * 128 // 32,), dtype=torch.uint8, requires_grad=False)))
        self.register_buffer('low_rank_weight', (torch.randint(1, 7, (self.out_features, 128 // 2), dtype=torch.uint8, requires_grad=False)))

        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None


    def set_low_rank_weight(self):
        pre_low_rank_weight = torch.rand(self.out_features, 128, dtype=torch.bfloat16, device=self.weight.device)
        self.low_rank_weight.data, self.SF_low_rank_weight.data = serq.simple_quantize_mxfp4(torch.abs(pre_low_rank_weight))
        # self.low_rank_weight.data = low_rank_weight
        # self.SF_low_rank_weight.data = SF_low_rank_weight
        del pre_low_rank_weight
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        assert type(x) == serq.PackedQuantizedTensor # Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x

        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        # out = serq.matmul(x, self.weight, scales_x, self.SF_B)
        
        out = serq.matmul(x[...,:64].contiguous(),self.low_rank_weight, scales_x, self.SF_low_rank_weight)
        out = serq.matmul_lowrank(x, self.weight, out, scales_x, self.SF_w)
        #out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return out + self.bias 
        else:
            return out 


    @staticmethod
    def from_float(module: SVDLinear):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        # To-do!
        
        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=module.weight_matrix.dtype)
        
        # main weight quantization
        assert module.weight.dtype == torch.bfloat16
        module.weight.cuda()
        weight_matrix, SF_w = serq.simple_quantize_mxfp4(module.weight)
        int_module.weight.copy_(weight_matrix.cpu())
        int_module.SF_w.copy_(SF_w.cpu())

        # low-rank weight quantization
        assert module.lora_R.dtype == torch.bfloat16
        module.lora_R.cuda()
        low_rank_weight, SF_low_rank_weight = serq.simple_quantize_mxfp4(module.lora_R)
        int_module.low_rank_weight.copy_(low_rank_weight.cpu())
        int_module.SF_low_rank_weight.copy_(SF_low_rank_weight.cpu())

        if module.bias is not None:
            int_module.bias.copy_(module.bias)
        
        return int_module