import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Callable



def _complex_gelu(input: Tensor) -> Tensor:
    return F.gelu(input.real).type(input.dtype) + 1.j*F.gelu(input.imag).type(input.dtype)


# Module about Fourier Neural Operator on Quantum Computing
# Input: f(batch, 16)
# Output: A(batch, 16)
class FNO_model(nn.Module):
    def __init__(self):
        super().__init__()        
        c_in = c_out = 1
        c_fft = 32    
        mode_fft = 16     
        # P_in, P_out
        self.P_in = mlp_linear(c_in, c_fft, n_layers=2, hidden_channels=256)
        self.P_out = mlp_linear(c_fft, c_out, n_layers=2, hidden_channels=256)
        # fourier space
        self.W_fft = nn.ModuleList()
        num_fourier_layers = 4
        for _ in range(num_fourier_layers):
            self.W_fft.append(mlp_linear(mode_fft * c_fft, mode_fft * c_fft, n_layers=2, 
                                         hidden_channels=1024, dtype=torch.cfloat))
        self.act_func = _complex_gelu
        ###

    def forward(self, x: Tensor) -> Tensor:        
        '''
        Input: x(batch, 16)
        Output: (batch, 16)
        '''
        if x.dim() == 2: x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = self.P_in(x)        
        # fourier space
        x = x.permute(0, 2, 1) # (b, c, fft)
        x_shape = x.shape
        for i in range(len(self.W_fft)):
            x_res = x
            x_fft = torch.fft.fft(x, dim=-1)            
            x_fft = self.W_fft[i](x_fft.reshape(x_shape[0], -1))
            x_fft = torch.fft.ifft(x.reshape(x_shape), dim=-1)
            x = x_fft.real + x_res
            if i < len(self.W_fft) - 1: 
                x = self.act_func(x)
        x = x.permute(0, 2, 1)  # (b, fft, c)
        x = self.P_out(x)
        x = x.squeeze(-1)
        return x
        







# MLP, linear or conv
class mlp_base(nn.Module):     
    def __init__(self, main_layer_type: Callable[[int, int], nn.Module], 
                 in_channels: int, out_channels: int, n_layers: int = 1, 
                 hidden_channels: int | None = None, non_linearity = F.gelu) -> None:
        super().__init__()
        in_features = in_channels
        out_features = in_channels if out_channels is None else out_channels
        if isinstance(hidden_channels, int) and hidden_channels > 0:
            num_layers = n_layers if n_layers > 2 else 2
            hidden_features = hidden_channels                 
        else:
            num_layers = 1
            hidden_features = 0    
        # main layers
        self.fcs = nn.ModuleList()
        # main layers          
        for i in range(num_layers):
            if i == 0 and i == (num_layers - 1):
                self.fcs.append(main_layer_type(in_features, out_features))
            elif i == 0:
                self.fcs.append(main_layer_type(in_features, hidden_features))
            elif i == (num_layers - 1):
                self.fcs.append(main_layer_type(hidden_features, out_features))                         
            else:
                self.fcs.append(main_layer_type(hidden_features, hidden_features))           
        # act
        self.non_linearity = non_linearity 
        ###
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:          
        num_layers = len(self.fcs)
        # main layers
        for i, fc in enumerate(self.fcs):
            x = fc(x)                   
            if i < num_layers - 1:
                x = self.non_linearity(x)            
        return x



class mlp_linear(mlp_base):   
    def __init__(self, in_channels: int, out_channels: int, n_layers: int = 1, 
                hidden_channels: int | None = None, non_linearity = F.gelu,
                dtype: torch.dtype = torch.float, use_bias: bool = True) -> None:
        # layer type
        main_layer_type = lambda c_in, c_out: nn.Linear(c_in, c_out, bias=use_bias, dtype=dtype)           
        # init
        super().__init__(main_layer_type, in_channels, out_channels, n_layers, 
                        hidden_channels, non_linearity)
        ###






     



        












