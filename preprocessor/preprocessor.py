import os
import random
import json

import tgt
import librosa
import numpy as np
from tqdm import tqdm

from audio import Audio
from text import grapheme_to_phoneme
from utils.tools import read_lexicon
from g2p_en import G2p

random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.frame_shift_sample = config["preprocessing"]["audio"]["frame_shift_sample"]
        self.clip_norm = config["preprocessing"]["mel"]["normalize"]

        self.g2p = G2p()
        self.audio_processor = Audio(config)
        self.lexicon = read_lexicon(config["path"]["lexicon_path"])

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in tqdm(os.listdir(os.path.join(self.in_dir, speaker))):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]

                ret = self.process_utterance(speaker, basename, self.clip_norm)
                if ret is None:
                    continue
                else:
                    info, n = ret
                out.append(info)

                n_frames += n

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        print(
            "Total time: {} hours".format(
                n_frames * self.frame_shift_sample / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename, clip_norm=False):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))

        # Compute mel-scale spectrogram from raw audio
        mel_spectrogram = self.audio_processor.get_mel_from_wav(wav_path, clip_norm=clip_norm)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Get phoneme
        phone = grapheme_to_phoneme(raw_text, self.g2p, self.lexicon)
        text = "{" + " ".join(phone) + "}"

        # Save files
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            mel_spectrogram.shape[1],
        )
