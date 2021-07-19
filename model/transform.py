import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LinearNorm, PositionalEncoding
from .attention import CrossAttentionBLK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BaseTransform, self).__init__()
        self.out_dim = out_dim
        self.log_scale_proj = LinearNorm(in_dim, self.out_dim,
                                        kernel_initializer='zeros')
        self.shift_proj = LinearNorm(in_dim, self.out_dim,
                                    kernel_initializer='zeros')

    def forward(self, inputs, condition_inputs, condition_lengths=None
             ):
        """
        :param inputs: xa inputs
        :param condition_inputs:
        :param condition_lengths:
        :return: tensor1: log_scale, tensor2: bias
        """
        raise NotImplementedError


class TransformerTransform(BaseTransform):
    def __init__(self, nblk, channels, embd_dim, attention_dim, attention_heads, temperature,
                 ffn_hidden, out_dim):
        super(TransformerTransform, self).__init__(in_dim=attention_dim, out_dim=out_dim)
        self.pos_emb_layer = PositionalEncoding()
        self.pos_weight = nn.Parameter(torch.tensor(1.0, device=device))
        self.pre_projection = LinearNorm(channels // 2, attention_dim)
        self.attentions = nn.ModuleList(
            [
                CrossAttentionBLK(input_dim=attention_dim,
                                memory_dim=embd_dim,
                                attention_dim=attention_dim,
                                attention_heads=attention_heads,
                                attention_temperature=temperature,
                                ffn_hidden=ffn_hidden)
                for i in range(nblk)
            ]
        )

    def forward(self, inputs, condition_inputs, condition_lengths=None,
             target_lengths=None):
        att_outs = self.pre_projection(inputs)
        max_time = att_outs.shape[1]
        dim = att_outs.shape[2]
        pos_embd = self.pos_emb_layer.positional_encoding(max_time, dim, device)
        att_outs += self.pos_weight * pos_embd
        for att in self.attentions:
            att_outs, _ = att(inputs=att_outs, memory=condition_inputs,
                              memory_lengths=condition_lengths,
                              query_lengths=target_lengths)
        log_scale = self.log_scale_proj(att_outs)
        shift = self.shift_proj(att_outs)
        return log_scale, shift
