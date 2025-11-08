import torch
from torch import nn
from torch.distributed.tensor import DTensor

class ObserverBase(nn.Module):
    def __init__(self, channels):
        super(ObserverBase, self).__init__()
        self.channels = channels
        if channels > 0:
            self.register_buffer('min_val', torch.zeros((1, 1, channels), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1, 1, channels), dtype=torch.float32))
        else:
            self.register_buffer('min_val', torch.zeros((1), dtype=torch.float32))
            self.register_buffer('max_val', torch.zeros((1), dtype=torch.float32))
        self.num_flag = 0


class MinMaxObserver(ObserverBase):
    def __init__(self, channels):
        super(MinMaxObserver, self).__init__(channels)

    def forward(self, x):
        if self.channels > 0:
            min_val = torch.min(x, 1, keepdim=True)[0]
            max_val = torch.max(x, 1, keepdim=True)[0]
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
        self.update_range(min_val, max_val)


class NormalMinMaxObserver(MinMaxObserver):
    def __init__(self, channels):
        super(NormalMinMaxObserver, self).__init__(channels)

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = torch.min(min_val, self.min_val)
            max_val_new = torch.max(max_val, self.max_val)

        if not isinstance(min_val_new, DTensor) and isinstance(self.min_val, DTensor):
            min_val_new = DTensor.from_local(min_val_new, self.min_val.device_mesh, self.min_val.placements, run_check=False)
        if not isinstance(max_val_new, DTensor) and isinstance(self.max_val, DTensor):
            max_val_new = DTensor.from_local(max_val_new, self.max_val.device_mesh, self.max_val.placements, run_check=False)

        with torch.no_grad():
            self.min_val.copy_(min_val_new)
            self.max_val.copy_(max_val_new)


class ActGroupObserver(nn.Module):
    def __init__(self, channels, gs):
        super(ActGroupObserver, self).__init__()
        self.gs = gs
        self.num_gr = channels // gs
        self.register_buffer('min_val', torch.zeros((1, 1, self.num_gr, 1), dtype=torch.float32))
        self.register_buffer('max_val', torch.zeros((1, 1, self.num_gr, 1), dtype=torch.float32))
        self.num_flag = 0

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = torch.min(min_val, self.min_val)
            max_val_new = torch.max(max_val, self.max_val)

        if not isinstance(min_val_new, DTensor) and isinstance(self.min_val, DTensor):
            min_val_new = DTensor.from_local(min_val_new, self.min_val.device_mesh, self.min_val.placements, run_check=False)
        if not isinstance(max_val_new, DTensor) and isinstance(self.max_val, DTensor):
            max_val_new = DTensor.from_local(max_val_new, self.max_val.device_mesh, self.max_val.placements, run_check=False)

        with torch.no_grad():
            self.min_val.copy_(min_val_new)
            self.max_val.copy_(max_val_new)

    def forward(self, x):
        act = x.reshape(x.shape[0], x.shape[1], self.num_gr, -1)

        min_val = torch.min(act, dim=3, keepdim=True)[0]
        min_val = torch.min(min_val, dim=1, keepdim=True)[0]
        max_val = torch.max(act, dim=3, keepdim=True)[0]
        max_val = torch.max(max_val, dim=1, keepdim=True)[0]
        
        self.update_range(min_val, max_val)
        


class ActChannelObserver(nn.Module):
    def __init__(self, channels):
        super(ActChannelObserver, self).__init__()
        self.register_buffer('min_val', torch.zeros((1, 1, channels), dtype=torch.float32))
        self.register_buffer('max_val', torch.zeros((1, 1, channels), dtype=torch.float32))
        self.num_flag = 0

    def update_range(self, min_val, max_val):
        # # ------ for parallel --------- #
        # if isinstance(min_val, DTensor):
        #     min_val = min_val.to_local()
        # if isinstance(max_val, DTensor):
        #     max_val = max_val.to_local()
            
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = torch.min(min_val, self.min_val)
            max_val_new = torch.max(max_val, self.max_val)

        with torch.no_grad():
            self.min_val.copy_(min_val_new)
            self.max_val.copy_(max_val_new)

    def forward(self, x):
        min_val = torch.min(x, dim=1, keepdim=True)[0]
        max_val = torch.max(x, dim=1, keepdim=True)[0]

        self.update_range(min_val, max_val)



class MovingAvgMinMaxObserver(MinMaxObserver):
    def __init__(self, channels, momentum=0.1):
        super(MovingAvgMinMaxObserver, self).__init__(channels)
        self.momentum = momentum

    def update_range(self, min_val, max_val):
        min_val = torch.reshape(min_val, self.min_val.shape)
        max_val = torch.reshape(max_val, self.max_val.shape)

        if not isinstance(min_val, DTensor):
            min_val = DTensor.from_local(min_val, self.min_val.device_mesh, self.min_val.placements, run_check=False)
        if not isinstance(max_val, DTensor):
            max_val = DTensor.from_local(max_val, self.max_val.device_mesh, self.max_val.placements, run_check=False)

        if self.num_flag == 0:
            min_val_new = min_val
            max_val_new = max_val
            self.num_flag += 1
        else:
            min_val_new = self.min_val + (min_val - self.min_val) * self.momentum
            max_val_new = self.max_val + (max_val - self.max_val) * self.momentum

        with torch.no_grad():
            self.min_val.copy_(min_val_new)
            self.max_val.copy_(max_val_new)