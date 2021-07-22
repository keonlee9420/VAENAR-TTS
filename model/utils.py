import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, activation=None,
                 use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias)
        # init weight
        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.linear.weight)
        elif kernel_initializer == 'zeros':
            nn.init.zeros_(self.linear.weight)
        # init bias
        if use_bias:
            if bias_initializer == 'zeros':
                nn.init.constant_(self.linear.bias, 0.0)
            else:
                raise NotImplementedError
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.activation(self.linear(x))
        return x


class ConvNorm(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=1, stride=1,
            padding=None, dilation=1, activation=None,
            use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(ConvNorm, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=use_bias
                              )

        # init weight
        if kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        elif kernel_initializer == 'zeros':
            nn.init.zeros_(self.conv.weight)

        # init bias
        if use_bias:
            if bias_initializer == 'zeros':
                nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PreNet(nn.Module):
    def __init__(self, in_features, units, drop_rate, activation):
        super(PreNet, self).__init__()
        self.dense1 = LinearNorm(
            in_features, units, activation=activation)
        self.dense2 = LinearNorm(
            units, units, activation=activation)
        self.dropout_layer = nn.Dropout(p=drop_rate)

    def forward(self, inputs):
        dense1_out = self.dense1(inputs)
        dense1_out = self.dropout_layer(dense1_out)
        dense2_out = self.dense2(dense1_out)
        dense2_out = self.dropout_layer(dense2_out)
        return dense2_out


class ConvPreNet(nn.Module):
    def __init__(self, nconv, hidden, conv_kernel, drop_rate,
                 activation=nn.ReLU(), bn_before_act=True):
        super(ConvPreNet, self).__init__()
        self.conv_stack = nn.ModuleList(
            [
                Conv1D(in_channels=hidden, out_channels=hidden, kernel_size=conv_kernel, activation=activation,
                       drop_rate=drop_rate, bn_before_act=bn_before_act)
                for i in range(nconv)
            ]
        )
        self.projection = LinearNorm(hidden, hidden)

    def forward(self, inputs, mask=None):
        conv_outs = inputs
        for conv in self.conv_stack:
            conv_outs = conv(conv_outs, mask)
        projections = self.projection(conv_outs)
        return projections


class FFN(nn.Module):
    def __init__(self, in_features, hidden1, hidden2):
        super(FFN, self).__init__()
        self.dense1 = LinearNorm(in_features, hidden1, activation=nn.ReLU())
        self.dense2 = LinearNorm(hidden1, hidden2, activation=None)
        self.layer_norm = nn.LayerNorm(hidden2)

    def forward(self, inputs, mask=None):
        dense1_outs = self.dense1(inputs)
        dense2_outs = self.dense2(dense1_outs)
        outs = dense2_outs + inputs
        outs = self.layer_norm(outs)
        if mask is not None:
            outs = outs.masked_fill(mask.unsqueeze(-1), 0.0)
        return outs


class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, drop_rate,
                 bn_before_act=False, strides=1):
        super(Conv1D, self).__init__()
        self.conv1d = ConvNorm(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=strides,
                               padding=int((kernel_size - 1) / 2),
                               dilation=1,
                               activation=None)
        self.activation = activation if activation is not None else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=drop_rate)
        self.bn_before_act = bn_before_act

    def forward(self, inputs, mask=None):
        conv_outs = inputs.contiguous().transpose(1, 2)
        conv_outs = self.conv1d(conv_outs)
        if self.bn_before_act:
            conv_outs = self.bn(conv_outs)
            conv_outs = self.activation(conv_outs)
        else:
            conv_outs = self.activation(conv_outs)
            conv_outs = self.bn(conv_outs)
        dropouts = self.dropout(conv_outs)
        dropouts = dropouts.contiguous().transpose(1, 2)
        if mask is not None:
            dropouts = dropouts.masked_fill(mask.unsqueeze(-1), 0.0)
        return dropouts


class PostNet(nn.Module):
    def __init__(self, n_conv, hidden, conv_filters, conv_kernel,
                 drop_rate):
        super(PostNet, self).__init__()
        activations = [nn.Tanh()] * (n_conv - 1) + [nn.Identity()]
        self.conv_stack = nn.ModuleList(
            [
                Conv1D(in_channels=hidden if i == 0 else conv_filters, out_channels=conv_filters,
                       kernel_size=conv_kernel,
                       activation=activations[i], drop_rate=drop_rate)
                for i in range(n_conv)
            ]
        )

    def forward(self, inputs, mask=None):
        conv_out = inputs
        for conv in self.conv_stack:
            conv_out = conv(conv_out, mask)
        return conv_out


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    @staticmethod
    def positional_encoding(len, dim, device, step=1.):
        """
        :param len: int scalar
        :param dim: int scalar
        :param device:
        :param step:
        :return: position embedding
        """
        pos_mat = torch.tile(
            (torch.arange(0, len, dtype=torch.float32, device=device) * step).unsqueeze(-1),
            [1, dim])
        dim_mat = torch.tile(
            torch.arange(0, dim, dtype=torch.float32, device=device).unsqueeze(0),
            [len, 1])
        dim_mat_int = dim_mat.type(torch.int32)
        pos_encoding = torch.where(  # [time, dims]
            torch.eq(torch.fmod(dim_mat_int, 2), 0),
            torch.sin(pos_mat / torch.pow(10000., dim_mat / float(dim))),
            torch.cos(pos_mat / torch.pow(10000., (dim_mat - 1) / float(dim))))
        return pos_encoding
