import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import Conv1D, ConvPreNet, PositionalEncoding
from .attention import SelfAttentionBLK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEncoder(nn.Module):
    def __init__(self, vocab_size, embd_dim):
        super(BaseEncoder, self).__init__()
        self.emb_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embd_dim,
                                        padding_idx=0)

    def forward(self, inputs, input_lengths=None):
        """
        :param inputs: text inputs, [batch, max_time]
        :param input_lengths: text inputs' lengths, [batch]
        :return: (tensor1, tensor2)
                tensor1: text encoding, [batch, max_time, hidden_size]
                tensor2: global state, i.e., final_time_state, [batch, hidden_size]
        """
        raise NotImplementedError


class TransformerEncoder(BaseEncoder):
    def __init__(self, vocab_size, embd_dim, pre_nconv, pre_hidden, pre_conv_kernel,
                 prenet_drop_rate, pre_activation, bn_before_act, pos_drop_rate, nblk,
                 attention_dim, attention_heads, attention_temperature, ffn_hidden):
        super(TransformerEncoder, self).__init__(vocab_size, embd_dim)
        self.pos_weight = nn.Parameter(torch.tensor(1.0, device=device))
        self.prenet = ConvPreNet(nconv=pre_nconv, hidden=pre_hidden,
                                 conv_kernel=pre_conv_kernel, drop_rate=prenet_drop_rate,
                                 activation=pre_activation, bn_before_act=bn_before_act)
        self.pe = PositionalEncoding()
        self.pe_dropout = nn.Dropout(p=pos_drop_rate)
        self.self_attentions = nn.ModuleList(
            [
                SelfAttentionBLK(
                    input_dim=pre_hidden, attention_dim=attention_dim,
                    attention_heads=attention_heads, attention_temperature=attention_temperature,
                    ffn_hidden=ffn_hidden)
                for i in range(nblk)
            ]
        )

    def forward(self, inputs, input_lengths=None, pos_step=1.0):
        # print('tracing back at text encoding')
        embs = self.emb_layer(inputs)
        prenet_outs = self.prenet(embs)
        max_time = prenet_outs.shape[1]
        dim = prenet_outs.shape[2]
        pos = self.pe.positional_encoding(max_time, dim, device, pos_step)
        pos_embs = prenet_outs + self.pos_weight * pos
        pos_embs = self.pe_dropout(pos_embs)
        att_outs = pos_embs
        for att in self.self_attentions:
            att_outs, alignments = att(
                inputs=att_outs, memory=att_outs, query_lengths=input_lengths,
                memory_lengths=input_lengths)
        return att_outs
