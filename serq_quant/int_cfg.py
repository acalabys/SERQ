import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--qphase', help='train/calib/validation', type=int, default=3)

parser.add_argument('--qnw', help='data bit-with', type=int, default=4)
parser.add_argument('--qna', help='data bit-with', type=int, default=4)
parser.add_argument('--qnq', help='data bit-with', type=int, default=16)
parser.add_argument('--qnk', help='data bit-with', type=int, default=16)
parser.add_argument('--qnv', help='data bit-with', type=int, default=16)

parser.add_argument('--kvquant', help='kv quant mode', type=int, default=0)
parser.add_argument('--asym', help='asym quant mode', action='store_true')
parser.add_argument('--mxfp4', help='mxfp4 quant mode', action='store_true')

opt, _ = parser.parse_known_args()

print("Set quantization phase: ", opt.qphase)
print("Set wegith quantization bit-width: ", opt.qnw)
print("Set activation quantization bit-width: ", opt.qna)
print("Set query quantization bit-width: ", opt.qnq)
print("Set key quantization bit-width: ", opt.qnk)
print("Set value quantization bit-width: ", opt.qnv)

# print("Set kv quant mode: ", opt.kvquant)
print("Set mxfp4 quant mode: ", opt.mxfp4)
print("Set asym quant mode: ", opt.asym)

class QInfo():
    def __init__(self, phase, data, n):
        self.phase = phase
        self.data = data
        self.n = n