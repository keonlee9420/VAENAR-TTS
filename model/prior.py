import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from numpy import pi as PI
from .glow import Glow
from utils.tools import get_mask_from_lengths


class Prior(nn.Module):
    """ P(z|x): prior that generate the latent variables conditioned on x
    """

    def __init__(self, channels):
        super(Prior, self).__init__()
        self.channels = channels

    def _initial_sample(self, targets_lengths, temperature=1.0):
        """
        :param targets_lengths: [batch,]
        :param temperature: standard deviation
        :return: initial samples with shape [batch_size, length, channels],
                 log-probabilities: [batch, ]
        """
        batch_size = targets_lengths.shape[0]
        length = torch.max(targets_lengths).long()
        epsilon = torch.normal(0.0, temperature, [batch_size, length, self.channels]).to(targets_lengths.device)

        logprobs = -0.5 * (epsilon ** 2 + math.log(2 * PI))
        seq_mask = get_mask_from_lengths(targets_lengths).unsqueeze(-1)  # [batch, max_time, 1]
        logprobs = torch.sum(seq_mask * logprobs, dim=[1, 2])  # [batch, ]
        return epsilon, logprobs

    def forward(self, inputs, targets_lengths, condition_lengths):
        """
        :param targets_lengths: [batch, ]
        :param inputs: condition_inputs
        :param condition_lengths:
        :return: tensor1: outputs, tensor2: log_probabilities
        """
        raise NotImplementedError

    def log_probability(self, z, condition_inputs, z_lengths=None, condition_lengths=None
                        ):
        """
        compute the log-probability of given latent variables, first run through the flow
        inversely to get the initial sample, then compute the
        :param z: latent variables
        :param condition_inputs: condition inputs
        :param z_lengths:
        :param condition_lengths:
        :return: the log-probability
        """
        raise NotImplementedError

    def init(self, *inputs, **kwargs):
        """
        Initiate the weights according to the initial input data
        :param inputs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError


class TransformerPrior(Prior):
    def __init__(self, n_blk, channels, n_transformer_blk, embd_dim, attention_dim,
                 attention_heads, temperature, ffn_hidden):
        super(TransformerPrior, self).__init__(channels)
        self.glow = Glow(n_blk, channels, n_transformer_blk, embd_dim, attention_dim,
                         attention_heads, temperature, ffn_hidden)

    def forward(self, targets_lengths, conditional_inputs: torch.Tensor, condition_lengths, temperature=1.0):
        # get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths, temperature=temperature)

        z = epsilon
        z, logdet = self.glow(z, conditional_inputs, targets_lengths, condition_lengths)
        logprobs += logdet
        return z, logprobs

    def log_probability(self, z, condition_inputs, z_lengths=None, condition_lengths=None):
        """
        :param z: [batch, max_time, dim]
        :param condition_inputs:
        :param z_lengths:
        :param condition_lengths:
        :return: log-probabilities of z, [batch]
        """
        epsilon, logdet = self.glow.inverse(z, condition_inputs, z_lengths, condition_lengths)

        logprobs = -0.5 * (epsilon ** 2 + math.log(2 * PI))
        max_time = z.shape[1]
        seq_mask = get_mask_from_lengths(z_lengths, max_time).unsqueeze(-1)  # [batch, max_time]
        logprobs = torch.sum(seq_mask * logprobs, dim=[1, 2])  # [batch, ]
        logprobs += logdet
        return logprobs

    def init(self, inputs: torch.Tensor, targets_lengths, condition_lengths):
        # get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths)

        z = epsilon
        z, logdet = self.glow.init(z, inputs, targets_lengths, condition_lengths)
        return z, logprobs
