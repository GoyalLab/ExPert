import torch


class ScaleByFactor:
    def __init__(self, factor: float = 10):
        self.factor = factor
    
    def __call__(self, x: torch.Tensor):
        return x * self.factor
