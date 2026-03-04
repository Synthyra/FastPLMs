import torch
import torch.nn as nn
import torch.nn.functional as F


def correction_fn(expansion_ratio: float, hidden_size: int, modulo: int = 256) -> int:
    """
    Compute intermediate size rounded up to nearest multiple of modulo.
    
    Used for feedforward layer sizing to ensure efficient tensor operations.
    
    Args:
        expansion_ratio: Multiplier for hidden size.
        hidden_size: Base hidden dimension.
        modulo: Round up to this multiple (default 256 for GPU efficiency).
        
    Returns:
        Intermediate size as int.
    """
    return int(((expansion_ratio * hidden_size) + modulo - 1) // modulo * modulo)


class SwiGLU_FFN(nn.Module):
    def __init__(self, hidden_size: int, expansion_ratio: float, dropout: float = 0.1, bias: bool = False):
        super(SwiGLU_FFN, self).__init__()
        corrected_dim = correction_fn(expansion_ratio, hidden_size)
        self.w1 = nn.Linear(hidden_size, corrected_dim, bias=bias)
        self.w2 = nn.Linear(corrected_dim, hidden_size, bias=bias)
        self.w3 = nn.Linear(hidden_size, corrected_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w3(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class ReLU2_FFN(nn.Module):
    def __init__(self, hidden_size: int, expansion_ratio: float, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        corrected_dim = correction_fn(expansion_ratio, hidden_size)
        self.w1 = nn.Linear(hidden_size, corrected_dim, bias=bias)
        self.w2 = nn.Linear(corrected_dim, hidden_size, bias=bias)
        self.w2.weight.data.zero_()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)).square())


class FFN(nn.Module):
    def __init__(self, hidden_size: int, expansion_ratio: float, dropout: float = 0.1, bias: bool = False, ffn_type: str = "swiglu"):
        super().__init__()
        if ffn_type == "swiglu":
            self.mlp = SwiGLU_FFN(hidden_size, expansion_ratio, dropout, bias)
        elif ffn_type == "relu2":
            self.mlp = ReLU2_FFN(hidden_size, expansion_ratio, dropout, bias)
        else:
            raise ValueError(f"Invalid FFN type: {ffn_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)