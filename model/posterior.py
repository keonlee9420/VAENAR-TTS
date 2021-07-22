import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from .utils import LinearNorm, PreNet, PositionalEncoding
from .attention import CrossAttentionBlock
from utils.tools import get_mask_from_lengths


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePosterior(nn.Module):
    """Encode the target sequence into latent distributions"""

    def __init__(self):
        super(BasePosterior, self).__init__()

    def forward(self, inputs, src_enc, src_lengths=None, target_lengths=None
                ):
        raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar, nsamples=1, random=True):
        """
        :param mu: [batch, max_time, dim]
        :param logvar: [batch, max_time, dim]
        :param nsamples: int
        :param random: whether sample from N(0, 1) or just use zeros
        :return: samples, noises, [batch, nsamples, max_time, dim]
        """
        # print('tracing back at posterior reparameterize')
        batch, max_time, dim = mu.shape
        std = torch.exp(0.5 * logvar)
        if random:
            eps = torch.normal(0.0, 1.0, [batch, nsamples, max_time, dim]).to(mu.device)
        else:
            eps = torch.zeros([batch, nsamples, max_time, dim], device=mu.device)
        samples = eps * std.unsqueeze(1) + mu.unsqueeze(1)
        return samples, eps

    @staticmethod
    def log_probability(mu, logvar, z=None, eps=None, seq_lengths=None, epsilon=1e-8):
        """
        :param mu: [batch, max_time, dim]
        :param logvar: [batch, max_time, dim]
        :param z: [batch, nsamples, max_time, dim]
        :param eps: [batch, nsamples, max_time, dim]
        :param seq_lengths: [batch, ]
        :param epsilon: small float number to avoid overflow
        :return: log probabilities, [batch, nsamples]
        """
        # print('tracing back at posterior log-probability')
        batch, max_time, dim = mu.shape

        # random noise
        # std = torch.exp(0.5 * logvar)
        normalized_samples = (eps if eps is not None
                              else (z - mu.unsqueeze(1))
                                   / (torch.exp(0.5 * logvar).unsqueeze(1) + epsilon))

        expanded_logvar = logvar.unsqueeze(1)
        # time_level_log_probs [batch, nsamples, max_time]
        time_level_log_probs = -0.5 * (float(dim) * math.log(2 * np.pi)
                                       + torch.sum(expanded_logvar + normalized_samples ** 2, dim=3))
        seq_mask = (get_mask_from_lengths(seq_lengths, max_time)
                    if seq_lengths is not None
                    else torch.ones([batch, max_time], device=mu.device))
        seq_mask = seq_mask.unsqueeze(1)  # [batch, 1, max_time]
        sample_level_log_probs = torch.sum(seq_mask * time_level_log_probs, dim=2)  # [batch, nsamples]
        return sample_level_log_probs

    def sample(self, inputs, src_enc, input_lengths, src_lengths,
               nsamples=1, random=True):
        """
        :param inputs: [batch, tgt_max_time, in_dim]
        :param src_enc: [batch, src_max_time, emb_dim]
        :param input_lengths: [batch, ]
        :param src_lengths: [batch, ]
        :param nsamples:
        :param random:
        :return:
        tensor1: samples from the posterior, [batch, nsamples, tgt_max_time, dim]
        tensor2: log-probabilities, [batch, nsamples]
        """
        raise NotImplementedError


class TransformerPosterior(BasePosterior):
    def __init__(self, num_mels, embd_dim, pre_hidden, pre_drop_rate, pre_activation,
                 pos_drop_rate, nblk, attention_dim, attention_heads,
                 temperature, ffn_hidden, latent_dim):
        super(TransformerPosterior, self).__init__()
        # self.pos_weight = nn.Parameter(torch.tensor(1.0, device=device))
        self.register_parameter("pos_weight", nn.Parameter(torch.tensor(1.0)))
        self.prenet = PreNet(in_features=num_mels, units=pre_hidden, drop_rate=pre_drop_rate,
                             activation=pre_activation)
        self.pe = PositionalEncoding()
        self.pe_dropout = nn.Dropout(p=pos_drop_rate)
        self.attentions = nn.ModuleList(
            [
                CrossAttentionBlock(input_dim=pre_hidden,
                                    memory_dim=embd_dim,
                                    attention_dim=attention_dim,
                                    attention_heads=attention_heads,
                                    attention_temperature=temperature,
                                    ffn_hidden=ffn_hidden)
                for i in range(nblk)
            ]
        )
        self.mu_projection = LinearNorm(attention_dim,
                                        latent_dim,
                                        kernel_initializer='none')
        self.logvar_projection = LinearNorm(attention_dim,
                                            latent_dim,
                                            kernel_initializer='none')

    def forward(self, inputs, src_enc, src_lengths=None, target_lengths=None):
        # print('tracing back at posterior call')
        prenet_outs = self.prenet(inputs)
        max_time = prenet_outs.shape[1]
        dim = prenet_outs.shape[2]
        pos = self.pe.positional_encoding(max_time, dim, inputs.device)
        pos_embs = prenet_outs + self.pos_weight * pos
        pos_embs = self.pe_dropout(pos_embs)
        att_outs = pos_embs
        for att in self.attentions:
            att_outs, alignments = att(
                inputs=att_outs, memory=src_enc, query_lengths=target_lengths,
                memory_lengths=src_lengths)

        # [batch, target_lengths, latent_dim]
        mu = self.mu_projection(att_outs)
        logvar = self.logvar_projection(att_outs)
        return mu, logvar, None

    def sample(self, inputs, src_enc, input_lengths, src_lengths,
               nsamples=1, random=True):
        mu, logvar, _ = self.forward(inputs, src_enc, input_lengths, src_lengths)
        samples, eps = self.reparameterize(mu, logvar, nsamples, random)
        log_probs = self.log_probability(mu, logvar, eps, input_lengths)
        return samples, log_probs
