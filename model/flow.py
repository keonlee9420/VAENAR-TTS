import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple
from .transform import TransformerTransform
from utils.tools import get_mask_from_lengths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseFlow(nn.Module):
    def __init__(self, inverse):
        super(BaseFlow, self).__init__()
        self.inverse = inverse

    def _forward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *inputs: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def _backward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def forward(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            *inputs: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        if self.inverse:
            return self._backward(*inputs, **kwargs)
        return self._forward(*inputs, **kwargs)

    def init(self, *inputs, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initiate the weights according to the initial input data
        :param inputs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def fwd_pass(self, inputs, *h, init=False, init_scale=1.0,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            inputs: Tensor
                The random variable before flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial y / \partial x`
            Then the density :math:`\log(p(y)) = \log(p(x)) - logdet`

        """
        if self.inverse:
            if init:
                raise RuntimeError(
                    'inverse flow shold be initialized with backward pass')
            else:
                return self._backward(inputs, *h, **kwargs)
        else:
            if init:
                return self.init(inputs, *h, init_scale=init_scale, **kwargs)
            else:
                return self._forward(inputs, *h, **kwargs)

    def bwd_pass(self, inputs, *h, init=False, init_scale=1.0,
                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param inputs: the random variable after the flow
        :param h: other conditional inputs
        :param init: bool, whether perform initialization or not
        :param init_scale: float (default: 1.0)
        :param kwargs:
        :return: x: the random variable before the flow,
                 log_det: the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`
        """
        if self.inverse:
            if init:
                return self.init(inputs, *h, init_scale=init_scale, **kwargs)
            else:
                return self._forward(inputs, *h, **kwargs)
        else:
            if init:
                raise RuntimeError(
                    'forward flow should be initialzed with forward pass')
            else:
                return self._backward(inputs, *h, **kwargs)


class InvertibleLinearFlow(BaseFlow):
    def __init__(self, channels, inverse):
        super(InvertibleLinearFlow, self).__init__(inverse)
        self.channels = channels
        w_init = np.linalg.qr(np.random.randn(channels, channels))[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init).type(torch.float32).to(device))

    def _forward(self, inputs, inputs_lengths=None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = torch.matmul(inputs, self.weight)
        logdet = torch.linalg.slogdet(
                self.weight.type(torch.float64))[1].type(torch.float32)
        if inputs_lengths is None:
            logdet = torch.ones(input_shape[0], device=device) * float(input_shape[1]) * logdet
        else:
            logdet = inputs_lengths.type(torch.float32) * logdet
        return outputs, logdet

    def _backward(self, inputs, inputs_lengths=None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = torch.matmul(inputs, torch.linalg.inv(self.weight))
        logdet = torch.linalg.slogdet(
                torch.linalg.inv(
                    self.weight.type(torch.float64)))[1].type(torch.float32)
        if inputs_lengths is None:
            logdet = torch.ones(input_shape[0], device=device) * float(input_shape[1]) * logdet
        else:
            logdet = inputs_lengths.type(torch.float32) * logdet
        return outputs, logdet

    def init(self, inputs, inputs_lengths=None):
        return self._forward(inputs, inputs_lengths)


class ActNormFlow(BaseFlow):
    def __init__(self, channels, inverse):
        super(ActNormFlow, self).__init__(inverse)
        self.channels = channels
        self.log_scale = nn.Parameter(torch.normal(0.0, 0.05, [self.channels, ]).to(device))
        self.bias = nn.Parameter(torch.zeros(self.channels, device=device))

    def _forward(self, inputs, input_lengths=None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = inputs * torch.exp(self.log_scale) + self.bias
        logdet = torch.sum(self.log_scale)
        if input_lengths is None:
            logdet = torch.ones(input_shape[0], device=device) * float(input_shape[1]) * logdet
        else:
            logdet = input_lengths.type(torch.float32) * logdet
        return outputs, logdet

    def _backward(self, inputs, input_lengths=None, epsilon=1e-8
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = inputs.shape
        outputs = (inputs - self.bias) / (torch.exp(self.log_scale) + epsilon)
        logdet = -torch.sum(self.log_scale)
        if input_lengths is None:
            logdet = torch.ones(input_shape[0], device=device) * float(input_shape[1]) * logdet
        else:
            logdet = input_lengths.type(torch.float32) * logdet
        return outputs, logdet

    def init(self, inputs, input_lengths=None, init_scale=1.0, epsilon=1e-8):
        _mean = torch.mean(inputs.view(-1, self.channels), dim=0)
        _std = torch.std(inputs.view(-1, self.channels), dim=0)
        self.log_scale.copy_(torch.log(init_scale / (_std + epsilon)))
        self.bias.copy_(-_mean / (_std + epsilon))
        return self._forward(inputs, input_lengths)


class TransformerCoupling(BaseFlow):
    def __init__(self, channels, inverse, nblk, embd_dim, attention_dim, attention_heads,
                 temperature, ffn_hidden, order='upper'):
        # assert channels % 2 == 0
        out_dim = channels // 2
        self.channels = channels
        super(TransformerCoupling, self).__init__(inverse)
        self.net = TransformerTransform(
            nblk=nblk, channels=channels, embd_dim=embd_dim, attention_dim=attention_dim, attention_heads=attention_heads,
            temperature=temperature, ffn_hidden=ffn_hidden, out_dim=out_dim)
        self.upper = (order == 'upper')

    @staticmethod
    def _split(inputs):
        return torch.split(inputs, split_size_or_sections=inputs.shape[-1] // 2, dim=-1)

    @staticmethod
    def _affine(inputs, scale, shift):
        return scale * inputs + shift

    @staticmethod
    def _inverse_affine(inputs, scale, shift, epsilon=1e-12):
        return (inputs - shift) / (scale + epsilon)

    def _forward(self, inputs, condition_inputs,
                 inputs_lengths=None, condition_lengths=None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert inputs.shape[-1] == self.channels
        lower_pt, upper_pt = self._split(inputs)
        z, zp = (lower_pt, upper_pt) if self.upper else (upper_pt, lower_pt)
        log_scale, shift = self.net(z, condition_inputs, condition_lengths,
                                    inputs_lengths)
        scale = torch.sigmoid(log_scale + 2.0)
        zp = self._affine(zp, scale, shift)
        inputs_max_time = inputs.shape[1]
        mask = (get_mask_from_lengths(inputs_lengths, inputs_max_time).unsqueeze(-1)
                if inputs_lengths is not None else torch.ones_like(log_scale))
        logdet = torch.sum(torch.log(scale) * mask, dim=[1, 2])  # [batch, ]
        outputs = torch.cat([z, zp], dim=-1) if self.upper else torch.cat([zp, z], dim=-1)
        return outputs, logdet

    def _backward(self, inputs, condition_inputs,
                  inputs_lengths=None, condition_lengths=None
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert inputs.shape[-1] == self.channels
        lower_pt, upper_pt = self._split(inputs)
        z, zp = (lower_pt, upper_pt) if self.upper else (upper_pt, lower_pt)
        log_scale, shift = self.net(z, condition_inputs, condition_lengths,
                                    inputs_lengths)
        scale = torch.sigmoid(log_scale + 2.0)
        zp = self._inverse_affine(zp, scale, shift)
        inputs_max_time = inputs.shape[1]
        mask = (get_mask_from_lengths(inputs_lengths, inputs_max_time).unsqueeze(-1)
                if inputs_lengths is not None else torch.ones_like(log_scale))
        log_det = -torch.sum(torch.log(scale) * mask, dim=[1, 2])  # [batch,]
        outputs = torch.cat([z, zp], dim=-1) if self.upper else torch.cat([zp, z], dim=-1)
        return outputs, log_det

    def init(self, inputs, condition_inputs, inputs_lengths=None,
             condition_lengths=None):
        return self._forward(
            inputs, condition_inputs, inputs_lengths, condition_lengths)
