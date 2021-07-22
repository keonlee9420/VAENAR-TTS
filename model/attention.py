import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from .utils import LinearNorm, FFN
from utils.tools import get_mask_from_lengths

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseAttention(nn.Module):
    def __init__(self, attention_dim):
        super(BaseAttention, self).__init__()
        self.attention_dim = attention_dim

    def forward(self, inputs, memory, memory_lengths, query_lengths):
        """
        :param inputs: query, [batch, q_time, q_dim]
        :param memory: [batch, m_time, m_dim]
        :param memory_lengths: [batch,]
        :param query_lengths: [batch,]
        :return: (tensor1, tensor2)
            tensor1: contexts, [batch, q_time, attention_dim]
            tensor2: alignments, probabilities, [batch, q_time, m_time]
        """
        raise NotImplementedError

    @staticmethod
    def _get_key_mask(batch_size, memory_max_time, query_max_time, memory_lengths, query_lengths, device):
        memory_lengths = (memory_lengths if memory_lengths is not None
                          else torch.ones(batch_size, dtype=torch.int32, device=device) * memory_max_time)
        memeory_mask = get_mask_from_lengths(memory_lengths, memory_max_time)
        memeory_mask = torch.tile(memeory_mask.unsqueeze(1),  # [batch, 1, m_max_time]
                               [1, query_max_time, 1])  # [batch, q_max_time, m_max_time]
        query_lengths = (query_lengths if query_lengths is not None
                         else torch.ones(batch_size, dtype=torch.int32, device=device) * query_max_time)
        query_mask = get_mask_from_lengths(query_lengths, query_max_time)  # [batch, q_max_time]
        query_mask = torch.tile(query_mask.unsqueeze(2),  # [batch, q_max_time, 1]
                             [1, 1, memory_max_time])  # [batch, q_max_time, m_max_time]
        length_mask = torch.logical_and(memeory_mask, query_mask)
        return length_mask


class MultiHeadScaledProductAttention(BaseAttention):
    def __init__(self, attention_dim, input_dim, memory_dim, num_head, temperature=1.0):
        assert attention_dim % num_head == 0
        super(MultiHeadScaledProductAttention, self).__init__(
            attention_dim=attention_dim)
        self.query_layer = LinearNorm(
            input_dim, attention_dim, use_bias=False)
        self.key_layer = LinearNorm(
            memory_dim, attention_dim, use_bias=False)
        self.value_layer = LinearNorm(
            memory_dim, attention_dim, use_bias=False)
        self.num_head = num_head
        self.temperature = temperature

    def _split_head(self, inputs):
        """
        :param inputs: [batch, time, dim]
        :return: [batch, num_head, time, dim // head]
        """
        batch, max_time, dim = inputs.shape
        reshaped = inputs.reshape(batch, max_time, self.num_head,
                               dim // self.num_head)
        # [batch, time, num_head, dim // head]
        transposed = reshaped.permute(0, 2, 1, 3)
        # [batch, num_head, time, dim // head]
        return transposed

    def _merge_head(self, inputs):
        """
        :param inputs: [batch, num_head, time, dim]
        :return: [batch, time, attention_dim]
        """
        batch, _, time, head_dim = inputs.shape
        transposed = inputs.permute(0, 2, 1, 3)
        # [batch, time, num_head, dim]
        reshaped = transposed.reshape(batch, time, self.num_head * head_dim)
        return reshaped

    def _get_key_mask(self, batch_size, memory_max_time, query_max_time,
                      memory_lengths, query_lengths, device):
        memory_lengths = (memory_lengths if memory_lengths is not None
                          else torch.ones(batch_size, dtype=torch.int32, device=device) * memory_max_time)
        memory_mask = get_mask_from_lengths(memory_lengths, memory_max_time)  # [batch, m_max_time]
        memory_mask = torch.tile(memory_mask.unsqueeze(1),  # [batch, 1, m_max_time]
                              [1, query_max_time, 1])  # [batch, q_max_time, m_max_time]
        query_lengths = (query_lengths if query_lengths is not None
                         else torch.ones(batch_size, dtype=torch.int32, device=device) * query_max_time)
        query_mask = get_mask_from_lengths(query_lengths, query_max_time)  # [batch, q_max_time]
        query_mask = torch.tile(query_mask.unsqueeze(2),  # [batch, q_max_time, 1]
                             [1, 1, memory_max_time])  # [batch, q_max_time, m_max_time]
        length_mask = torch.logical_and(memory_mask, query_mask)
        length_mask = torch.tile(length_mask.unsqueeze(1),
                              [1, self.num_head, 1, 1])
        # [batch, num_head, q_max_time, m_max_time]
        return length_mask

    @staticmethod
    def _get_causal_mask(logits):
        causal_mask = torch.tril(torch.ones(logits.shape, dtype=torch.bool, device=logits.device))
        return causal_mask

    def forward(self, inputs, memory, memory_lengths=None, query_lengths=None, causality=None):
        queries = self.query_layer(inputs)  # [batch, Tq, D]
        keys = self.key_layer(memory)  # [batch, Tk, D]
        values = self.value_layer(memory)  # [batch, Tk, Dv]
        headed_queries = self._split_head(queries)  # [batch, num_head, Tq, head_dim]
        headed_keys = self._split_head(keys)  # [batch, num_head, Tk, head_dim]
        headed_values = self._split_head(values)  # [batch, num_head, Tk, head_dim]
        logits = torch.matmul(headed_queries,
                                  headed_keys.transpose(-2, -1))  # [batch, num_head, Tq, Tk]
        logits = logits / math.sqrt(
            float(self.attention_dim // self.num_head))  # scale
        logits = logits / self.temperature  # temperature
        # apply mask
        batch_size = memory.shape[0]
        memory_max_time = memory.shape[1]
        query_max_time = inputs.shape[1]
        length_mask = self._get_key_mask(
            batch_size, memory_max_time, query_max_time, memory_lengths, query_lengths, inputs.device)
        if causality:
            causal_mask = self._get_causal_mask(logits)
            length_mask = torch.logical_and(length_mask, causal_mask)
        # [batch, num_head, q_max_time, m_max_time]
        paddings = torch.ones_like(logits, dtype=torch.float32) * (-2. ** 32 + 1)
        logits = torch.where(length_mask, logits, paddings)
        alignments = torch.softmax(logits, dim=3)  # [batch, num_head, Tq, Tk]
        contexts = torch.matmul(alignments, headed_values)
        # [batch, num_head, Tq, head_dim]
        contexts = self._merge_head(contexts)  # [batch, Tq, attention_dim]
        return contexts, alignments


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, attention_dim, attention_heads, attention_temperature,
                 ffn_hidden):
        super(SelfAttentionBlock, self).__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.attention = MultiHeadScaledProductAttention(attention_dim=attention_dim,
                                                         input_dim=input_dim,
                                                         memory_dim=input_dim,
                                                         num_head=attention_heads,
                                                         temperature=attention_temperature)
        self.att_proj = LinearNorm(attention_dim + input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.ffn = FFN(in_features=input_dim, hidden1=ffn_hidden, hidden2=input_dim)

    def forward(self, inputs, memory, query_lengths, memory_lengths, causality=None):
        att_outs, alignments = self.attention(inputs=inputs, memory=memory,
                                              query_lengths=query_lengths,
                                              memory_lengths=memory_lengths,
                                              causality=causality)
        contexts = torch.cat([inputs, att_outs], dim=-1)
        att_outs = self.att_proj(contexts)
        att_outs = self.layer_norm(inputs + att_outs)
        ffn_outs = self.ffn(att_outs)
        return ffn_outs, alignments


class CrossAttentionBlock(nn.Module):
    def __init__(self, input_dim, memory_dim, attention_dim, attention_heads, attention_temperature,
                 ffn_hidden, name=None):
        super(CrossAttentionBlock, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.self_attention = MultiHeadScaledProductAttention(
            attention_dim=attention_dim, input_dim=input_dim, memory_dim=input_dim, num_head=attention_heads,
            temperature=attention_temperature)
        self.att_proj1 = LinearNorm(attention_dim + input_dim, input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.cross_attention = MultiHeadScaledProductAttention(
            attention_dim=attention_dim, input_dim=input_dim, memory_dim=memory_dim, num_head=attention_heads,
            temperature=attention_temperature)
        self.att_proj2 = LinearNorm(attention_dim * 2, attention_dim)
        self.layer_norm2 = nn.LayerNorm(attention_dim)
        self.ffn = FFN(in_features=attention_dim, hidden1=ffn_hidden, hidden2=attention_dim)

    def forward(self, inputs, memory, query_lengths, memory_lengths):
        self_att_outs, self_ali = self.self_attention(
            inputs=inputs, memory=inputs, query_lengths=query_lengths,
            memory_lengths=query_lengths, causality=True)
        contexts = torch.cat([inputs, self_att_outs], dim=-1)
        self_att_outs = self.att_proj1(contexts)
        self_att_outs = self.layer_norm1(self_att_outs + inputs)
        att_outs, cross_ali = self.cross_attention(
            inputs=self_att_outs, memory=memory, query_lengths=query_lengths,
            memory_lengths=memory_lengths, causality=False)
        contexts = torch.cat([self_att_outs, att_outs], dim=-1)
        att_outs = self.att_proj2(contexts)
        att_outs = self.layer_norm2(att_outs + self_att_outs)
        ffn_outs = self.ffn(att_outs)
        return ffn_outs, cross_ali
