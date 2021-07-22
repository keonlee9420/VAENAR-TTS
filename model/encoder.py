import torch
import torch.nn as nn
from torch.nn import functional as F
from math import sqrt

from .utils import ConvPreNet, PositionalEncoding
from .attention import SelfAttentionBlock


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoder(nn.Module):
    def __init__(self, n_symbols, embedding_dim, pre_nconv, pre_hidden, pre_conv_kernel,
                 prenet_drop_rate, pre_activation, bn_before_act, pos_drop_rate, n_blocks,
                 attention_dim, attention_heads, attention_temperature, ffn_hidden):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_symbols,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        std = sqrt(2.0 / (n_symbols + embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        self.prenet = ConvPreNet(nconv=pre_nconv, hidden=pre_hidden,
                                 conv_kernel=pre_conv_kernel, drop_rate=prenet_drop_rate,
                                 activation=pre_activation, bn_before_act=bn_before_act)

        self.register_parameter("pos_weight", nn.Parameter(torch.tensor(1.0)))
        self.pe = PositionalEncoding()
        self.pe_dropout = nn.Dropout(p=pos_drop_rate)

        self.self_attentions = nn.ModuleList(
            [
                SelfAttentionBlock(
                    input_dim=pre_hidden, attention_dim=attention_dim,
                    attention_heads=attention_heads, attention_temperature=attention_temperature,
                    ffn_hidden=ffn_hidden)
                for i in range(n_blocks)
            ]
        )

    def forward(self, inputs, input_lengths=None, pos_step=1.0):
        # print('tracing back at text encoding')
        embs = self.embedding(inputs)
        prenet_outs = self.prenet(embs)
        max_time, prenet_dim = prenet_outs.size(1), prenet_outs.size(2)
        pos = self.pe.positional_encoding(max_time, prenet_dim, inputs.device, pos_step)
        pos_embs = prenet_outs + self.pos_weight * pos
        pos_embs = self.pe_dropout(pos_embs)
        att_outs = pos_embs
        for att in self.self_attentions:
            att_outs, alignments = att(
                inputs=att_outs, memory=att_outs, query_lengths=input_lengths,
                memory_lengths=input_lengths)
        return att_outs
