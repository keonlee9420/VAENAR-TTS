import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from numpy import pi as PI
from .flow import InvertibleLinearFlow, ActNormFlow, TransformerCoupling
from utils.tools import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePrior(nn.Module):
    """ P(z|x): prior that generate the latent variables conditioned on x
    """

    def __init__(self, channels):
        super(BasePrior, self).__init__()
        self.channels = channels

    def forward(self, inputs, targets_lengths, condition_lengths
             ):
        """
        :param targets_lengths: [batch, ]
        :param inputs: condition_inputs
        :param condition_lengths:
        :return: tensor1: outputs, tensor2: log_probabilities
        """
        raise NotImplementedError

    def _initial_sample(self, targets_lengths, temperature=1.0):
        """
        :param targets_lengths: [batch,]
        :param temperature: standard deviation
        :return: initial samples with shape [batch_size, length, channels],
                 log-probabilities: [batch, ]
        """
        batch_size = targets_lengths.shape[0]
        length = torch.max(targets_lengths).type(torch.int32)
        epsilon = torch.normal(0.0, temperature, [batch_size, length, self.channels]).to(device)
        logprobs = -0.5 * (math.log(2. * PI) + epsilon ** 2)
        seq_mask = get_mask_from_lengths(targets_lengths).unsqueeze(-1)  # [batch, max_time, 1]
        logprobs = torch.sum(seq_mask * logprobs, dim=[1, 2])  # [batch, ]
        return epsilon, logprobs

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

    def sample(self, targets_lengths, n_samples, condition_inputs, condition_lengths=None
               ):
        """
        :param targets_lengths:
        :param n_samples:
        :param condition_inputs:
        :param condition_lengths:
        :return: tensor1: samples: [batch, n_samples, max_lengths, dim]
                 tensor2: log-probabilities: [batch, n_samples]
        """
        raise NotImplementedError


class TransformerPrior(BasePrior):
    def __init__(self, n_blk, channels, n_transformer_blk, embd_dim, attention_dim,
                 attention_heads, temperature, ffn_hidden, inverse=False):
        super(TransformerPrior, self).__init__(channels)
        orders = ['upper', 'lower']
        self.actnorms = nn.ModuleList(
            [
                ActNormFlow(channels, inverse)
                for i in range(n_blk)
            ]
        )
        self.linears = nn.ModuleList(
            [
                InvertibleLinearFlow(channels, inverse)
                for i in range(n_blk)
            ]
        )
        self.affine_couplings = nn.ModuleList(
            [
                TransformerCoupling(channels=channels, inverse=inverse,
                                    nblk=n_transformer_blk,
                                    embd_dim=embd_dim,
                                    attention_dim=attention_dim,
                                    attention_heads=attention_heads,
                                    temperature=temperature,
                                    ffn_hidden=ffn_hidden,
                                    order=orders[i % 2])
                for i in range(n_blk)
            ]
        )

    def forward(self, inputs, targets_lengths, condition_lengths, temperature=1.0
             ):
        # 1. get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths, temperature=temperature)
        z = epsilon
        for _, (actnorm, linear, affine_coupling) in enumerate(zip(self.actnorms, self.linears, self.affine_couplings)):
            z, logdet = actnorm(z, targets_lengths)
            logprobs -= logdet
            z, logdet = linear(z, targets_lengths)
            logprobs -= logdet
            z, logdet = affine_coupling(inputs=z, condition_inputs=inputs,
                                        inputs_lengths=targets_lengths,
                                        condition_lengths=condition_lengths)
            logprobs -= logdet
        return z, logprobs

    def log_probability(self, z, condition_inputs, z_lengths=None,
                        condition_lengths=None
                        ):
        """
        :param z: [batch, max_time, dim]
        :param condition_inputs:
        :param z_lengths:
        :param condition_lengths:
        :return: log-probabilities of z, [batch]
        """
        # print('tracing back at prior log-probability')
        epsilon = z
        batch_size = z.shape[0]
        max_time = z.shape[1]
        accum_logdet = torch.zeros([batch_size, ], dtype=torch.float32, device=device)
        for _, (actnorm, linear, affine_coupling) in enumerate(zip(self.actnorms, self.linears, self.affine_couplings)):
            epsilon, logdet = affine_coupling.bwd_pass(inputs=epsilon,
                                                       condition_inputs=condition_inputs,
                                                       inputs_lengths=z_lengths,
                                                       condition_lengths=condition_lengths)
            accum_logdet += logdet
            epsilon, logdet = linear.bwd_pass(epsilon, z_lengths)
            accum_logdet += logdet
            epsilon, logdet = actnorm.bwd_pass(epsilon, z_lengths)
            accum_logdet += logdet
        logprobs = -0.5 * (math.log(2. * PI) + epsilon ** 2)
        seq_mask = get_mask_from_lengths(z_lengths, max_time).unsqueeze(-1)  # [batch, max_time]
        logprobs = torch.sum(seq_mask * logprobs, dim=[1, 2])  # [batch, ]
        logprobs += accum_logdet
        return logprobs

    def sample(self, targets_lengths, condition_inputs, condition_lengths=None, temperature=1.0):
        # 1. get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths, temperature=temperature)  # [batch*n_samples, ]
        z = epsilon
        for _, (actnorm, linear, affine_coupling) in enumerate(zip(self.actnorms, self.linears, self.affine_couplings)):
            z, logdet = actnorm(z, targets_lengths)
            logprobs -= logdet
            z, logdet = linear(z, targets_lengths)
            logprobs -= logdet
            z, logdet = affine_coupling.fwd_pass(inputs=z, condition_inputs=condition_inputs,
                                                 inputs_lengths=targets_lengths,
                                                 condition_lengths=condition_lengths)
            logprobs -= logdet
        return z, logprobs

    def init(self, conditions, targets_lengths, condition_lengths):
        # 1. get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths)
        z = epsilon
        for _, (actnorm, linear, affine_coupling) in enumerate(zip(self.actnorms, self.linears, self.affine_couplings)):
            z, logdet = actnorm.init(z, targets_lengths)
            logprobs -= logdet
            z, logdet = linear(z, targets_lengths)
            logprobs -= logdet
            z, logdet = affine_coupling.init(inputs=z, condition_inputs=conditions,
                                             inputs_lengths=targets_lengths,
                                             condition_lengths=condition_lengths)
            logprobs -= logdet
        return z, logprobs