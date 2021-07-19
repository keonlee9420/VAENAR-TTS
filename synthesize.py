import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p

from audio import Audio
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples, read_lexicon
from dataset import TextDataset
from text import grapheme_to_phoneme, text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_english(text, preprocess_config):
    g2p = G2p()
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = grapheme_to_phoneme(text, g2p, lexicon)
    phones = "{" + " ".join(phones) + "}"

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, audio_processor, batchs, temperature):
    preprocess_config, model_config, train_config = configs

    final_reduction_factor = model_config["common"]["final_reduction_factor"]

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            t, t_l = batch[3], batch[4]
            text_pos_step = model.mel_text_len_ratio / float(final_reduction_factor)
            text_embd = model.text_encoder(t, t_l, pos_step=text_pos_step)
            predicted_lengths = model.length_predictor(
                text_embd.detach(), t_l)
            predicted_m_l = predicted_lengths.type(torch.int32)
            reduced_pred_ml = (predicted_m_l + 80 + final_reduction_factor - 1
                            ) // final_reduction_factor
            prior_latents, prior_logprobs = model.prior.sample(
                reduced_pred_ml, text_embd, t_l, temperature=temperature)
            _, prior_dec_outs, prior_dec_alignments = model.decoder(
                prior_latents, text_embd, reduced_pred_ml, t_l)

            synth_samples(
                batch,
                prior_dec_outs,
                predicted_m_l + 80,
                reduced_pred_ml,
                t_l,
                prior_dec_alignments,
                vocoder,
                audio_processor,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument('--temperature', type=float, default=0.)
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    audio_processor = Audio(preprocess_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    synthesize(model, args.restore_step, configs, vocoder, audio_processor, batchs, args.temperature)
