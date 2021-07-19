import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LinearNorm, PostNet
from .attention import CrossAttentionBLK
from utils.tools import get_mask_from_lengths


class BaseDecoder(nn.Module):
    """ P(y|x,z): decode target sequence from latent variables conditioned by x
    """

    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, inputs, text_embd, z_lengths=None, text_lengths=None,
             targets=None):
        """
        :param inputs: latent representations, [batch, max_audio_time, z_hidden]
        :param text_embd: text encodings, [batch, max_text_time, T_emb_hidden]
        :param z_lengths: [batch, ]
        :param text_lengths: [batch, ]
        :param targets: [batch, max_audio_time, out_dim]
        :return: tensor1: reconstructed acoustic features, tensor2: alignments
        """
        raise NotImplementedError

    @staticmethod
    def _compute_l1_loss(reconstructed, targets, lengths=None):
        if lengths is not None:
            max_time = targets.shape[1]
            seq_mask = get_mask_from_lengths(lengths, max_time)
            l1_loss = torch.mean(
                torch.sum(
                    torch.mean(
                        torch.abs(reconstructed - targets),
                        dim=-1) * seq_mask,
                    dim=-1) / lengths.type(torch.float32))
        else:
            l1_loss = F.l1_loss(reconstructed, targets)
        return l1_loss

    @staticmethod
    def _compute_l2_loss(reconstructed, targets, lengths=None):
        if lengths is not None:
            max_time = targets.shape[1]
            seq_mask = get_mask_from_lengths(lengths, max_time)
            l2_loss = torch.mean(
                torch.sum(
                    torch.mean(
                        torch.square(reconstructed - targets),
                        dim=-1) * seq_mask,
                    dim=-1) / lengths.type(torch.float32))
        else:
            l2_loss = F.mse_loss(reconstructed, targets)
        return l2_loss


class TransformerDecoder(BaseDecoder):
    def __init__(self, nblk, embd_dim, attention_dim, attention_heads,
                 temperature, ffn_hidden, post_n_conv, post_conv_filters,
                 post_conv_kernel, post_drop_rate, latent_dim, out_dim, max_reduction_factor):
        super(TransformerDecoder, self).__init__()
        self.max_reduction_factor = max_reduction_factor
        self.out_dim = out_dim
        self.pre_projection = LinearNorm(latent_dim, attention_dim)
        self.attentions = nn.ModuleList(
            [
                CrossAttentionBLK(input_dim=attention_dim,
                                memory_dim=embd_dim,
                                attention_dim=attention_dim,
                                attention_heads=attention_heads,
                                attention_temperature=temperature,
                                ffn_hidden=ffn_hidden, name='decoder-attention-{}'.format(i))
                for i in range(nblk)
            ]
        )
        self.out_projection = LinearNorm(attention_dim, out_dim * self.max_reduction_factor)
        self.postnet = PostNet(n_conv=post_n_conv, hidden=out_dim, conv_filters=post_conv_filters,
                               conv_kernel=post_conv_kernel, drop_rate=post_drop_rate)
        self.residual_projection = LinearNorm(post_conv_filters, out_dim)

    def forward(self, inputs, text_embd, z_lengths=None, text_lengths=None, reduction_factor=2):
        # print('Tracing back at Self-attention decoder')
        # shape info
        batch_size = inputs.shape[0]
        max_len = inputs.shape[1]
        att_outs = self.pre_projection(inputs)
        alignemnts = {}
        for att in self.attentions:
            att_outs, ali = att(
                inputs=att_outs, memory=text_embd, query_lengths=z_lengths,
                memory_lengths=text_lengths)
            alignemnts[att.name] = ali
        initial_outs = self.out_projection(att_outs)[:, :, : reduction_factor * self.out_dim]
        initial_outs = initial_outs.reshape(batch_size, max_len * reduction_factor, self.out_dim)
        residual = self.postnet(initial_outs)
        residual = self.residual_projection(residual)
        outputs = residual + initial_outs
        return initial_outs, outputs, alignemnts
