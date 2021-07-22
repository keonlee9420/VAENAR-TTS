import torch
import torch.nn as nn

from .flow import InvertibleLinearFlow, ActNormFlow, TransformerCoupling, Flow


class GlowBlock(Flow):
    def __init__(self, channels, n_transformer_blk, embd_dim, attention_dim,
                 attention_heads, temperature, ffn_hidden, order):
        super(GlowBlock, self).__init__()
        self.actnorm = ActNormFlow(channels)
        self.linear = InvertibleLinearFlow(channels)
        self.affine_coupling = TransformerCoupling(channels=channels,
                                                   nblk=n_transformer_blk,
                                                   embd_dim=embd_dim,
                                                   attention_dim=attention_dim,
                                                   attention_heads=attention_heads,
                                                   temperature=temperature,
                                                   ffn_hidden=ffn_hidden,
                                                   order=order)

    def forward(self, z, inputs: torch.Tensor, targets_lengths, condition_lengths):
        total_logdet = torch.zeros([z.size(0), ], device=z.device)

        z, logdet = self.actnorm(z, targets_lengths)
        total_logdet += logdet
        z, logdet = self.linear(z, targets_lengths)
        total_logdet += logdet
        z, logdet = self.affine_coupling(inputs=z, condition_inputs=inputs,
                                         inputs_lengths=targets_lengths,
                                         condition_lengths=condition_lengths)
        total_logdet += logdet
        return z, total_logdet

    def inverse(self, z, inputs: torch.Tensor, targets_lengths, condition_lengths):
        total_logdet = torch.zeros([z.size(0), ], device=z.device)

        # reverse order
        z, logdet = self.affine_coupling.inverse(inputs=z, condition_inputs=inputs,
                                                 inputs_lengths=targets_lengths,
                                                 condition_lengths=condition_lengths)
        total_logdet += logdet
        z, logdet = self.linear.inverse(z, targets_lengths)
        total_logdet += logdet
        z, logdet = self.actnorm.inverse(z, targets_lengths)
        total_logdet += logdet

        return z, total_logdet

    def init(self, z, inputs: torch.Tensor, targets_lengths, condition_lengths):
        total_logdet = torch.zeros([z.size(0), ], device=z.device)

        z, logdet = self.actnorm.init(z, targets_lengths)
        total_logdet += logdet
        z, logdet = self.linear.init(z, targets_lengths)
        total_logdet += logdet
        z, logdet = self.affine_coupling.init(inputs=z, condition_inputs=inputs,
                                              inputs_lengths=targets_lengths,
                                              condition_lengths=condition_lengths)
        total_logdet += logdet
        return z, total_logdet


class Glow(Flow):
    def __init__(self, n_blocks, channels, n_transformer_blk, embd_dim, attention_dim,
                 attention_heads, temperature, ffn_hidden):
        super(Glow, self).__init__()
        orders = ['upper', 'lower']
        self.flows = nn.ModuleList([GlowBlock(channels, n_transformer_blk, embd_dim, attention_dim,
                                              attention_heads, temperature, ffn_hidden, orders[i % 2])
                                    for i in range(n_blocks)])

    def forward(self, z, inputs: torch.Tensor, targets_lengths, condition_lengths):
        total_logdet = torch.zeros([z.size(0), ], device=z.device)
        for flow in self.flows:
            z, logdet = flow(z, inputs, targets_lengths, condition_lengths)
            total_logdet += logdet
        return z, total_logdet

    def inverse(self, z, inputs: torch.Tensor, targets_lengths, condition_lengths):
        total_logdet = torch.zeros([z.size(0), ], device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, logdet = self.flows[i].inverse(z, inputs, targets_lengths, condition_lengths)
            total_logdet += logdet
        return z, total_logdet

    def init(self, z, inputs: torch.Tensor, targets_lengths, condition_lengths):
        total_logdet = torch.zeros([z.size(0), ], device=z.device)
        for flow in self.flows:
            z, logdet = flow.init(z, inputs, targets_lengths, condition_lengths)
            total_logdet += logdet
        return z, total_logdet
