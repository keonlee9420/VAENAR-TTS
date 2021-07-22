import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .posterior import TransformerPosterior
from .decoder import TransformerDecoder
from .prior import TransformerPrior
from .length_predictor import DenseLengthPredictor

from utils.tools import get_mask_from_lengths
from text.symbols import symbols


class VAENAR(nn.Module):
    """ VAENAR-TTS """

    def __init__(self, preprocess_config, model_config):
        super(VAENAR, self).__init__()
        self.model_config = model_config
        self.n_sample = model_config["common"]["num_samples"]
        self.mel_text_len_ratio = model_config["common"]["mel_text_len_ratio"]
        self.max_reduction_factor = model_config["common"]["max_reduction_factor"]

        self.text_encoder = TransformerEncoder(
            n_symbols=len(symbols) + 1,
            embedding_dim=model_config["transformer"]["encoder"]["embd_dim"],
            pre_nconv=model_config["transformer"]["encoder"]["n_conv"],
            pre_hidden=model_config["transformer"]["encoder"]["pre_hidden"],
            pre_conv_kernel=model_config["transformer"]["encoder"]["conv_kernel"],
            pre_activation=self._get_activation(model_config["transformer"]["encoder"]["pre_activation"]),
            prenet_drop_rate=model_config["transformer"]["encoder"]["pre_drop_rate"],
            bn_before_act=model_config["transformer"]["encoder"]["bn_before_act"],
            pos_drop_rate=model_config["transformer"]["encoder"]["pos_drop_rate"],
            n_blocks=model_config["transformer"]["encoder"]["n_blk"],
            attention_dim=model_config["transformer"]["encoder"]["attention_dim"],
            attention_heads=model_config["transformer"]["encoder"]["attention_heads"],
            attention_temperature=model_config["transformer"]["encoder"]["attention_temperature"],
            ffn_hidden=model_config["transformer"]["encoder"]["ffn_hidden"], )
        self.decoder = TransformerDecoder(
            nblk=model_config["transformer"]["decoder"]["nblk"],
            embd_dim=model_config["transformer"]["encoder"]["embd_dim"],
            attention_dim=model_config["transformer"]["decoder"]["attention_dim"],
            attention_heads=model_config["transformer"]["decoder"]["attention_heads"],
            temperature=model_config["transformer"]["decoder"]["attention_temperature"],
            ffn_hidden=model_config["transformer"]["decoder"]["ffn_hidden"],
            post_n_conv=model_config["transformer"]["decoder"]["post_n_conv"],
            post_conv_filters=model_config["transformer"]["decoder"]["post_conv_filters"],
            post_conv_kernel=model_config["transformer"]["decoder"]["post_conv_kernel"],
            post_drop_rate=model_config["transformer"]["decoder"]["post_drop_rate"],
            latent_dim=model_config["common"]["latent_dim"],
            out_dim=model_config["common"]["output_dim"],
            max_reduction_factor=model_config["common"]["max_reduction_factor"])
        self.length_predictor = DenseLengthPredictor(
            embd_dim=model_config["transformer"]["encoder"]["embd_dim"],
            activation=self._get_activation(model_config["length_predictor"]["dense"]["activation"]))
        self.posterior = TransformerPosterior(
            num_mels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            embd_dim=model_config["transformer"]["encoder"]["embd_dim"],
            pre_hidden=model_config["transformer"]["posterior"]["pre_hidden"],
            pos_drop_rate=model_config["transformer"]["posterior"]["pos_drop_rate"],
            pre_drop_rate=model_config["transformer"]["posterior"]["pre_drop_rate"],
            pre_activation=self._get_activation(model_config["transformer"]["posterior"]["pre_activation"]),
            nblk=model_config["transformer"]["posterior"]["nblk"],
            attention_dim=model_config["transformer"]["posterior"]["attention_dim"],
            attention_heads=model_config["transformer"]["posterior"]["attention_heads"],
            temperature=model_config["transformer"]["posterior"]["temperature"],
            ffn_hidden=model_config["transformer"]["posterior"]["ffn_hidden"],
            latent_dim=model_config["common"]["latent_dim"])
        self.prior = TransformerPrior(
            n_blk=model_config["transformer"]["prior"]["n_blk"],
            channels=model_config["common"]["latent_dim"],
            n_transformer_blk=model_config["transformer"]["prior"]["n_transformer_blk"],
            embd_dim=model_config["transformer"]["encoder"]["embd_dim"],
            attention_dim=model_config["transformer"]["prior"]["attention_dim"],
            attention_heads=model_config["transformer"]["prior"]["attention_heads"],
            temperature=model_config["transformer"]["prior"]["temperature"],
            ffn_hidden=model_config["transformer"]["prior"]["ffn_hidden"]
        )

    @staticmethod
    def _get_activation(activation):
        if activation == "relu":
            return nn.ReLU()
        return None

    def _compute_l2_loss(self, reconstructed, targets, lengths=None, reduce=False):
        max_time = reconstructed.shape[1]
        dim = reconstructed.shape[2]
        r = reconstructed.view(-1, self.n_sample, max_time, dim)
        t = targets.view(-1, self.n_sample, max_time, dim)
        if lengths is not None:
            seq_mask = get_mask_from_lengths(lengths, max_time)
            seq_mask = seq_mask.view(-1, self.n_sample, max_time)
            reshaped_lens = lengths.view(-1, self.n_sample)
            l2_loss = torch.mean(
                torch.sum(
                    torch.mean(torch.square(r - t), dim=-1) * seq_mask,
                    dim=-1) / reshaped_lens.type(torch.float32),
                dim=-1)
        else:
            l2_loss = torch.mean(torch.square(r - t), dim=[1, 2, 3])
        if reduce:
            return torch.mean(l2_loss)
        else:
            return l2_loss

    @staticmethod
    def _kl_divergence(p, q, reduce=None):
        kl = torch.mean((p - q), dim=1)
        if reduce:
            return torch.mean(kl)
        else:
            return kl
        # kl = F.kl_div(p, F.softmax(q, dim=1))
        # return kl

    @staticmethod
    def _length_l2_loss(predicted_lengths, target_lengths, reduce=False):
        log_tgt_lengths = torch.log(target_lengths.type(torch.float32))
        log_pre_lengths = torch.log(predicted_lengths)
        if reduce:
            return torch.mean(torch.square(log_pre_lengths - log_tgt_lengths))
        else:
            return torch.square(log_pre_lengths - log_tgt_lengths)

    def forward(
            self,
            speakers,
            inputs,
            text_lengths,
            max_src_len,
            mel_targets=None,
            mel_lengths=None,
            max_mel_len=None,
            reduction_factor=2,
            reduce_loss=False,
    ):
        """
        :param speakers: speaker inputs, [batch, ]
        :param inputs: text inputs, [batch, text_max_time]
        :param text_lengths: [batch, ]
        :param max_src_len: int
        :param mel_targets: [batch, mel_max_time, mel_dim]
        :param mel_lengths: [batch, ]
        :param max_mel_len: int
        :param reduce_loss: bool
        :return: predicted mel: [batch, mel_max_time, mel_dim]
                 loss: float32
        """
        # print('tracing back at FlowTacotron.call')
        # shape info
        batch_size = mel_targets.shape[0]
        mel_max_len = mel_targets.shape[1]
        text_max_len = inputs.shape[1]
        # reduce the mels
        reduced_mels = mel_targets[:, ::reduction_factor, :]
        reduced_mel_lens = torch.div((mel_lengths + reduction_factor - 1), reduction_factor, rounding_mode='trunc')
        reduced_mel_max_len = reduced_mels.shape[1]

        # text encoding
        text_pos_step = self.mel_text_len_ratio / float(reduction_factor)
        text_embd = self.text_encoder(
            inputs, text_lengths, pos_step=text_pos_step)
        predicted_lengths = self.length_predictor(
            text_embd.detach(), text_lengths)
        length_loss = self._length_l2_loss(
            predicted_lengths, mel_lengths, reduce=reduce_loss)
        logvar, mu, post_alignments = self.posterior(reduced_mels, text_embd,
                                                     src_lengths=text_lengths,
                                                     target_lengths=reduced_mel_lens)

        # prepare batch
        # samples, eps: [batch, n_sample, mel_max_time, dim]
        samples, eps = self.posterior.reparameterize(mu, logvar, self.n_sample)
        # [batch, n_sample]
        posterior_logprobs = self.posterior.log_probability(mu, logvar, eps=eps, seq_lengths=reduced_mel_lens)

        # [batch*n_sample, mel_max_len, dim]
        batched_samples = samples.view(batch_size * self.n_sample, reduced_mel_max_len, -1)
        # [batch*n_sample, text_max_len, dim]
        batched_text_embd = torch.tile(
            text_embd.unsqueeze(1),
            [1, self.n_sample, 1, 1]).view(batch_size * self.n_sample, text_max_len, -1)
        batched_mel_targets = torch.tile(
            mel_targets.unsqueeze(1),
            [1, self.n_sample, 1, 1]).view(batch_size * self.n_sample, mel_max_len, -1)
        # [batch*n_sample, ]
        batched_mel_lengths = torch.tile(
            mel_lengths.unsqueeze(1),
            [1, self.n_sample]).view(-1)
        # [batch*n_sample, ]
        batched_r_mel_lengths = torch.tile(
            reduced_mel_lens.unsqueeze(1),
            [1, self.n_sample]).view(-1)
        # [batch*n_sample, ]
        batched_text_lengths = torch.tile(
            text_lengths.unsqueeze(1),
            [1, self.n_sample]).view(-1)

        # decoding
        decoded_initial, decoded_outs, dec_alignments = self.decoder(
            batched_samples, batched_text_embd, batched_r_mel_lengths,
            batched_text_lengths, reduction_factor=reduction_factor)
        decoded_initial = decoded_initial[:, :mel_max_len, :]
        decoded_outs = decoded_outs[:, :mel_max_len, :]
        initial_l2_loss = self._compute_l2_loss(decoded_initial, batched_mel_targets,
                                                batched_mel_lengths, reduce_loss)
        l2_loss = self._compute_l2_loss(decoded_outs, batched_mel_targets,
                                        batched_mel_lengths, reduce_loss)
        l2_loss += initial_l2_loss
        # [batch*n_sample, ]
        prior_logprobs = self.prior.log_probability(z=batched_samples,
                                                    condition_inputs=batched_text_embd,
                                                    z_lengths=batched_r_mel_lengths,
                                                    condition_lengths=batched_text_lengths)
        prior_logprobs = prior_logprobs.view(batch_size, self.n_sample)

        kl_divergence = self._kl_divergence(posterior_logprobs, prior_logprobs, reduce_loss)

        return (decoded_outs, l2_loss, kl_divergence, length_loss, dec_alignments, reduced_mel_lens,
                posterior_logprobs, prior_logprobs)

    def inference(self, inputs, text_lengths, reduction_factor=2):
        text_pos_step = self.mel_text_len_ratio / float(reduction_factor)
        text_embd = self.text_encoder(inputs, text_lengths, pos_step=text_pos_step)
        predicted_mel_lengths = (self.length_predictor(text_embd, text_lengths) + 80).long()
        reduced_mel_lens = (predicted_mel_lengths + reduction_factor - 1) // reduction_factor

        prior_latents, prior_logprobs = self.prior(reduced_mel_lens, text_embd, text_lengths)

        _, predicted_mel, dec_alignments = self.decoder(
            inputs=prior_latents, text_embd=text_embd, z_lengths=reduced_mel_lens,
            text_lengths=text_lengths, reduction_factor=reduction_factor)
        return predicted_mel, predicted_mel_lengths, reduced_mel_lens, dec_alignments, prior_logprobs

    def init(self, text_inputs, mel_lengths, text_lengths=None):
        text_pos_step = self.mel_text_len_ratio / float(self.max_reduction_factor)
        text_embd = self.text_encoder(text_inputs, text_lengths, pos_step=text_pos_step)
        reduced_mel_lens = (mel_lengths + self.max_reduction_factor - 1) // self.max_reduction_factor

        prior_latents, prior_logprobs = self.prior.init(inputs=text_embd,
                                                        targets_lengths=reduced_mel_lens,
                                                        condition_lengths=text_lengths)
        _, predicted_mel, _ = self.decoder(inputs=prior_latents,
                                           text_embd=text_embd,
                                           z_lengths=reduced_mel_lens,
                                           text_lengths=text_lengths,
                                           reduction_factor=self.max_reduction_factor)
        return predicted_mel
