import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
        model,
        step,
        configs,
        reduction_factor,
        length_weight,
        kl_weight,
        logger=None,
        vocoder=None,
        audio_processor=None,
        losses_len=4,
        device="cuda:0"):
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=True, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Evaluation
    loss_sums = [0 for _ in range(losses_len)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                (predictions, mel_l2, kl_divergence, length_l2, dec_alignments, reduced_mel_lens, *_) = model(
                    *(batch[2:]),
                    reduce_loss=True,
                    reduction_factor=reduction_factor
                )

                # Cal Loss
                total_loss = mel_l2 + length_weight * length_l2 + kl_weight * kl_divergence
                losses = list([total_loss, mel_l2, kl_divergence, length_l2])

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, KLD Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, attn_figs, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            predictions,
            dec_alignments,
            reduced_mel_lens,
            vocoder,
            audio_processor,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        for attn_idx, attn_fig in enumerate(attn_figs):
            log(
                logger,
                fig=attn_fig,
                tag="Validation_dec_attn_{}/step_{}_{}".format(attn_idx, step, tag),
            )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message
