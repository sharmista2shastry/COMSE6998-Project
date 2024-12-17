import torch
import math


def get_slopes_power_of_2(n):
    start = 2.0 ** (-2.0 ** -(math.log2(n) - 3))
    return [start * (2.0 ** (i / 2.0)) for i in range(n)]


def get_alibi_slopes(n_heads, device='cpu'):
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        slopes += [slopes[-1] * 2.0 ** ((i + 1) / 2.0) for i in range(n_heads - closest_power_of_2)]
    return torch.tensor(slopes, dtype=torch.float32, device=device)
