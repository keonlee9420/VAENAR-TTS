import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LinearNorm, Conv1D
from utils.tools import get_mask_from_lengths


class DenseLengthPredictor(nn.Module):
    def __init__(self, embd_dim, activation):
        super(DenseLengthPredictor, self).__init__()
        self.projection = LinearNorm(embd_dim, 1, activation=activation)

    def forward(self, inputs, input_lengths, training=None):
        proj_outs = self.projection(inputs)
        mask = get_mask_from_lengths(input_lengths, inputs.shape[1]).unsqueeze(-1)
        target_lengths = torch.sum((torch.exp(proj_outs) * mask).squeeze(-1), dim=-1)
        return target_lengths
