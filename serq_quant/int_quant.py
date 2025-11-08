import numpy as np
import torch
from typing import Optional
from torch.nn import functional as F
from torch.autograd import Function
from torch import nn
from torch.autograd import Function, Variable

from .observers import NormalMinMaxObserver, ActChannelObserver, ActGroupObserver
from .int_cfg import opt, QInfo

class act_quantizer(nn.Module):
    def __init__(
            self, 
            channels,
            qinfo,
            gs: Optional[int] = None, 
            device="cuda"
        ):
        super(act_quantizer, self).__init__()
        self.qinfo = qinfo
        self.device = device
        self.gs = gs
        self.smooth_observer = ActChannelObserver(channels)
        self.register_buffer('smooth_factor', torch.zeros_like(self.smooth_observer.max_val, dtype=torch.float32))

        if self.gs:
            self.observer = ActGroupObserver(channels, self.gs)
        else:
            self.observer = NormalMinMaxObserver(channels)
        

        self.register_buffer('scale', torch.ones_like((self.observer.max_val), dtype=torch.float32))

    def update_quant_params(self):
        print("update_quant_param_access")
        print('before scale', self.scale)
        print("QInfo data:", self.qinfo.data)

        quant_range = 2**(self.qinfo.n-1) - 1
        data_range = torch.max(torch.abs(self.observer.max_val), torch.abs(self.observer.min_val))
        self.scale = data_range / quant_range

        if (data_range == 0).any():
            self.scale[self.scale == 0] = 1
            print("There is a zero-value in scale so it is initialized to 1.")
            raise RuntimeError("There is a zero-value in scale so it is initialized to 1.")

        print('updated scale', self.scale)
        print("QInfo data:", self.qinfo.data)

    def update_smooth_factor(self):
        print("update_smooth_factor_access")
        print('before smooth_factor', self.smooth_factor)
        print("QInfo data:", self.qinfo.data)

        self.smooth_factor = torch.max(torch.abs(self.smooth_observer.max_val), torch.abs(self.smooth_observer.min_val))

        if (self.smooth_factor == 0).any():
            self.smooth_factor[self.smooth_factor == 0] = 1
            print("There is a zero-value in smooth_factor so it is initialized to 1.")
            raise RuntimeError("There is a zero-value in smooth_factor so it is initialized to 1.")
        print("QInfo data:", self.qinfo.data)
        print('updated smooth_factor', self.smooth_factor)


    def forward(self, x):
        if self.qinfo.phase == 0: # baseline 
            return x

        elif self.qinfo.phase == 1: # For smooth factor
            self.smooth_observer(x)
            return x

        # elif self.qinfo.phase == 2: # For scale factor
        #     # Dynamic quantization
        #     if opt.asym:
        #         # out = AsymPerGroupQuant.apply(x.squeeze(dim=(0)), opt.qna, 128)
        #         out = PerRowAsymQuant.apply(x.squeeze(dim=(0)), opt.qna)
        #     else:
        #         # out = PerGroupQuant.apply(x.squeeze(dim=(0)), opt.qna, 128)
        #         out = PerRowQuant.apply(x.squeeze(dim=(0)), opt.qna)
        #     out = out.unsqueeze(dim=(0))
        #     print(f"qphase2 exit")
        #     return out

        elif self.qinfo.phase == 3: # For Inference
            
            if opt.mxfp4:
                out = MXFP4.apply(x.squeeze(dim=(0)), 32)
            else:
                # Dynamic quantization
                if opt.asym:
                    out = AsymPerGroupQuant.apply(x.squeeze(dim=(0)), opt.qna, 128)
                else:
                    out = PerGroupQuant.apply(x.squeeze(dim=(0)), opt.qna, 128)
            out = out.unsqueeze(dim=(0))
    
            return out


class SVDLinear(nn.Linear):
    def __init__(
            self, 
            ic, 
            oc, 
            rank:int = 64,
            num_outlier:int = 128,
            bias=False, 
            is_qa=True
        ):
        super().__init__(ic, oc, bias)
        self.is_qa = is_qa
        self.rank = rank
        self.num_outlier = num_outlier
        
        self.qinfoa = QInfo(phase=opt.qphase, data='act', n=opt.qna)
        if is_qa:
            self.quantA = act_quantizer(ic, qinfo=self.qinfoa, gs=32)

        # self.lora_L = nn.Parameter(torch.zeros(oc, rank, dtype=torch.float16), requires_grad=False)   
        # self.lora_R = nn.Parameter(torch.zeros(rank, ic, dtype=torch.float16), requires_grad=False)
        self.lora_R = nn.Parameter(torch.zeros(oc, self.num_outlier, dtype=self.weight.dtype), requires_grad=False)

        self.register_buffer("mask", torch.zeros(ic, dtype=bool))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("order", torch.zeros(ic, dtype=torch.int32))


    def init_lora(self):
        ### Low-Matrix Initialization ###
 
        ### 1. Reorder & scaling weight ###
        w = self.weight.to(torch.float32)
        w = w * self.quantA.smooth_factor.squeeze(dim=(0)).to(self.weight.dtype)
        w = w[:, self.order]
        outlier_w = w[:, :self.num_outlier]

        ### 2. Quantize outlier weight ###
        R = PerRowQuant.apply(outlier_w, 4)
        Q = outlier_w - R
        R = R.to(self.weight.dtype)

        ### 3. Upate residual weight ###
        w[:, :self.num_outlier] = Q
        w = w.to(self.weight.dtype)

        ### 4. Update params ###
        with torch.no_grad():
            self.lora_R.data.copy_(R)
            self.weight.data.copy_(w)

        self.initialized.copy_(torch.tensor(True, dtype=torch.bool))

        # print(f"R shape: {self.lora_R.shape}")

    # Decompose only outlier & only one matrix
    def serq_quant(self):
        ### Low-Matrix Initialization ###

        ### 1. Reordering & scaling weight ###
        w = self.weight.to(torch.float32)
        w = w * self.quantA.smooth_factor.squeeze(dim=(0)).to(self.weight.dtype)
        w = w[:, self.order]
        outlier_w = w[:, :self.num_outlier]

        ### 2. Quantize outlier weight ###
        # R = PerRowQuant.apply(outlier_w, 4)
        R = MXFP4.apply(outlier_w, 32)
        Q = outlier_w - R
        R = R.to(self.weight.dtype)

        ### 3. Quantize weight ###
        w[:, :self.num_outlier] = Q
        # w = PerGroupQuant.apply(w, 4, 128)
        w = MXFP4.apply(w, 32)
        w = w.to(self.weight.dtype)

        ### 4. Update params ###
        with torch.no_grad():
            self.lora_R.data.copy_(R)
            self.weight.data.copy_(w)

        self.initialized.copy_(torch.tensor(True, dtype=torch.bool))


    def forward(self, x):
        if self.qinfoa.phase == 0:
            return F.linear(x, self.weight, self.bias)
            
        # qphase 1: smooth factor 계산
        if self.qinfoa.phase == 1:
            if self.is_qa:
                aquant = self.quantA(x)
            else:
                aquant = x
            out = F.linear(aquant, self.weight, self.bias)
            return out

        # # qphase 2: activation scale factor 계산
        # elif self.qinfoa.phase == 2:
        #     if self.is_qa:
        #         aquant = self.quantA(x)
        #     else:
        #         aquant = x
        #     aquant = aquant.to(self.weight.dtype)
        #     out = F.linear(aquant, self.weight, self.bias)
        #     return out

        # qphase 777: for GPTQ
        elif self.qinfoa.phase == 777:
            aquant = x.to(self.weight.dtype)
            out = F.linear(aquant, self.weight, self.bias)
            out = out + aquant[:, :, :self.num_outlier] @ self.lora_R.T
            return out

        # qphase 3: 실제 양자화 수행
        elif self.qinfoa.phase == 3:
            if self.is_qa:
                aquant = x / self.quantA.smooth_factor
                aquant = aquant[:,:,self.order]
                aquant = self.quantA(aquant)
            else:
                aquant = x
            aquant = aquant.to(self.weight.dtype)

            if self.initialized:
                out = F.linear(aquant, self.weight, self.bias)
                out = out + aquant[:, :, :self.num_outlier] @ self.lora_R.T
            else:
                out = F.linear(aquant, self.weight, self.bias)

            return out



class MXFP4(Function):
    """
        group은 row-wise로 묶임
    """
    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        gr: int = 32
    ):
        with torch.no_grad():
            data = data.to(torch.float32)
            sign_data = data.sign()
            abs_data = data.abs()
            rows, cols = abs_data.shape
            abs_data = abs_data.flatten()
            abs_data = abs_data.reshape(-1, gr) # (num_gr, gr)
            emax_elem = 2
            eps = torch.finfo(torch.float32).eps

            ### Calculate shared exponent ###
            gr_max = abs_data.max(dim=1, keepdim=True)[0] + eps
            shared_e = torch.floor(torch.log2(gr_max)) - emax_elem
            X = (2**shared_e)

            ### Normalization ###
            norm_data = abs_data / X

            ### Exponent ###
            e = torch.floor(torch.log2(norm_data))  # (~2)
            is_sub = e < 0
            e = torch.clamp(e, 0, 2)

            ### Mantissa ###
            m = ((norm_data / (2**e)) - 1) * 2
            m = m.round()

            carry = (m == 2)
            satu = carry & (e == 2)
            n_i = carry & (~satu)

            # normal case
            e = torch.where(n_i, e + 1, e)
            m = torch.where(n_i, torch.zeros_like(m), m)

            # saturation case
            m = torch.where(satu, torch.ones_like(m), m)

            m = m.clamp(0, 1)
            m = m * 0.5 + 1

            ### Subnormal ###
            m_sub = norm_data[is_sub] * 2
            m_sub = m_sub.round()

            carry_sub = (m_sub == 2)

            m_sub = torch.where(carry_sub, torch.ones_like(m_sub), m_sub.clamp(0, 1) * 0.5)

            m[is_sub] = m_sub
            e = torch.where(is_sub, torch.zeros_like(e), e)

            ### Encoding & Decoding ###
            encoded = (2**e) * m * X

            abs_data_encoded = encoded.reshape(rows, -1)
            result = abs_data_encoded * sign_data

            return result


class PerRowQuant(Function):
    @staticmethod
    def forward(ctx, data, num_bits: int = 8):
        with torch.no_grad():
            # 데이터 전처리: float32로 변환 및 부호 추출
            data = data.to(torch.float32)
            sign = torch.sign(data)
            data_abs = torch.abs(data)
            quant_range = 2**(num_bits-1) - 1
            
            # per-row scale 계산 (각 row마다 최댓값)
            data_range = torch.max(data_abs, dim=1, keepdim=True)[0]
            scale = data_range / quant_range
            if torch.any(torch.isnan(scale)):
                print("scale is nan")
            elif torch.any(scale == 0):
                print("scale is zero")
            
            # 양자화 (IntQuant 함수 구현)
            # data / scale를 반올림하고 클램프
            data_abs = IntQuant.apply(data_abs, scale, 0, num_bits, False, True) 
            
            # dequantize: 다시 실수 값으로 변환하고 원래 부호를 곱함
            output = (data_abs * scale) * sign
        
            return output


class PerRowAsymQuant(Function):
    @staticmethod
    def forward(ctx, data, num_bits: int = 8):
        with torch.no_grad():
            # 데이터 전처리: float32로 변환 및 부호 추출
            data = data.to(torch.float32)
            quant_range = 2**(num_bits) - 1
            
            # calculate scale factor
            data_max  = torch.amax(data, dim=1, keepdim=True)
            data_min  = torch.amin(data, dim=1, keepdim=True)
            tmp = (data_max==0) & (data_min==0)
            data_max[tmp] = 1
            data_min[tmp] = -1
            scale = (data_max - data_min) / quant_range
            zero = torch.round(-data_min / scale)

            # fake quant
            data = IntQuant.apply(data, scale, zero, num_bits, False, False)
            
            # DeQuant
            data = (data-zero) * scale
        
            return data


class PerGroupQuant(Function):
    @staticmethod
    def forward(ctx, data, num_bits: int = 8, gs: int = 128):
        """
            data는 2차원 텐서여야 함
            group은 row-wise
        """
        with torch.no_grad():
            ### Data preprocessing ###
            data = data.to(torch.float32)
            sign = torch.sign(data)
            data = torch.abs(data)

            R, C = data.shape
            G = C // gs  # 그룹 수
            quant_range = 2**(num_bits - 1) - 1

            data = data.reshape(R, G, gs).permute(1, 0, 2)

            ### Calculate scale factor ###
            data_range = data.amax(dim=2, keepdim=True)
            scale = data_range / quant_range    

            ### Fake Quant ###
            data = IntQuant.apply(data, scale, 0, num_bits, False, True) 
            data = data * scale                    

            ### Reconstruction ###
            data = data.permute(1, 0, 2).reshape(R, C)

            return data * sign
        

class AsymPerGroupQuant(Function):
    @staticmethod
    def forward(ctx, data, num_bits: int = 8, gs: int = 32):
        with torch.no_grad():
            data = data.to(torch.float32)

            # 1. set parameters
            R, C = data.shape
            G = C // gs  # 그룹 수
            quant_range = 2**(num_bits) - 1

            # 2. reshape
            data = data.reshape(R, G, gs).permute(1, 0, 2) # (num_gr, rows, gs)

            # 3. calculate scale factor
            data_max = data.amax(dim=2, keepdim=True) # (num_gr, rows, 1)
            data_min = data.amin(dim=2, keepdim=True) # (num_gr, rows, 1)
            tmp = (data_max==0) & (data_min==0)
            data_max[tmp] = 1
            data_min[tmp] = -1
            scale = (data_max - data_min) / quant_range
            zero = torch.round(-data_min / scale)

            # 4. fake quant
            data = IntQuant.apply(data, scale, zero, num_bits, False, False)

            # 5. dequant
            data = (data-zero) * scale

            # 6. reconstruct
            data = data.permute(1, 0, 2).reshape(R, C)
            return data
            
            
class IntQuant(Function):
    @staticmethod
    def forward(ctx, x, S, Z, N, is_stochastic: bool = False, sym: bool = True):
        if sym:
            level = 2**(N - 1) - 1
            out = x / S
            if is_stochastic:
                out = IntQuant.stochastic_rounding(out)
            else:
                out = torch.round(out)
            out = torch.clamp(out, max=level)
        else:
            level = 2**(N) - 1
            out = x / S
            if is_stochastic:
                out = IntQuant.stochastic_rounding(out)
            else:
                out = torch.round(out)
            out = torch.clamp(out+Z, min=0, max=level)
        return out


    @staticmethod
    def stochastic_rounding(x):
        q = torch.floor(x)
        delta = x - q
        r = torch.rand_like(delta, dtype=x.dtype, device=x.device)
        out = q + (r < delta).float()
        return out
