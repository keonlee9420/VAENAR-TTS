import torch
import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile


class Audio:
    def __init__(self, config):
        super(Audio, self).__init__()
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        self.pre_emphasize_value = config["preprocessing"]["audio"]["preemphasize"]
        self.ref_level_db = config["preprocessing"]["audio"]["ref_level_db"]
        self.num_freq = config["preprocessing"]["audio"]["num_freq"]
        self.frame_length_sample = config["preprocessing"]["audio"]["frame_length_sample"]
        self.frame_shift_sample = config["preprocessing"]["audio"]["frame_shift_sample"]
        self.center = config["preprocessing"]["audio"]["center"]
        self.griffin_lim_iters = config["preprocessing"]["audio"]["griffin_lim_iters"]
        self.num_mels = config["preprocessing"]["mel"]["n_mel_channels"]
        self.min_mel_freq = config["preprocessing"]["mel"]["min_mel_freq"]
        self.max_mel_freq = config["preprocessing"]["mel"]["max_mel_freq"]
        self.max_abs_value = config["preprocessing"]["mel"]["max_abs_value"]
        self.min_level_db = config["preprocessing"]["mel"]["min_level_db"]
        self.power = config["preprocessing"]["mel"]["power"]

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sampling_rate)[0]

    def save_wav(self, path, wav):
        wav *= self.max_wav_value / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, self.sampling_rate, wav.astype(np.int16))
        return

    def _stft_parameters(self):
        n_fft = (self.num_freq - 1) * 2
        hop_length = self.frame_shift_sample
        win_length = self.frame_length_sample
        return n_fft, hop_length, win_length

    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        if len(y.shape) == 1:  # [time_steps]
            return librosa.stft(y=y, n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                center=self.center)
        elif len(y.shape) == 2:  # [batch_size, time_steps]
            if y.shape[0] == 1:  # batch_size=1
                return np.expand_dims(librosa.stft(y=y[0], n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   win_length=win_length,
                                                   center=self.center),
                                      axis=0)
            else:  # batch_size > 1
                spec_list = list()
                for wav in y:
                    spec_list.append(librosa.stft(y=wav, n_fft=n_fft,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  center=self.center))
                return np.concatenate(spec_list, axis=0)
        else:
            raise Exception('Wav dimension error in stft function!')

    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        if len(y.shape) == 2:  # spectrogram shape: [n_frame, n_fft]
            return librosa.istft(y, hop_length=hop_length,
                                 win_length=win_length,
                                 center=self.center)
        elif len(y.shape) == 3:  # spectrogram shape: [batch_size, n_frame, n_fft]
            if y.shape[0] == 1:  # batch_size = 1
                return np.expand_dims(librosa.istft(y[0],
                                                    hop_length=hop_length,
                                                    win_length=win_length,
                                                    center=self.center),
                                      axis=0)
            else:  # batch_size > 1
                wav_list = list()
                for spec in y:
                    wav_list.append(librosa.istft(spec,
                                                  hop_length=hop_length,
                                                  win_length=win_length,
                                                  center=self.center))
                    return np.concatenate(wav_list, axis=0)
        else:
            raise Exception('Spectrogram dimension error in istft function!')

    @staticmethod
    def _amp_to_db(x):
        return 20 * np.log10(np.maximum(1e-5, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def _build_mel_basis(self):
        n_fft = (self.num_freq - 1) * 2
        return librosa.filters.mel(
            self.sampling_rate,
            n_fft=n_fft,
            n_mels=self.num_mels,
            fmin=self.min_mel_freq,
            fmax=self.max_mel_freq)

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def melspectrogram(self, y, clip_norm=True):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        if clip_norm:
            S = self._normalize(S)
        return S

    def get_mel_from_wav(self, audio_path, clip_norm=True):
        wav_arr = self.load_wav(audio_path)
        wav_arr = self.preemphasize(wav_arr)
        mels = self.melspectrogram(wav_arr, clip_norm=clip_norm)
        return mels

    def _mel_to_linear(self, mel_spectrogram):
        _inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        linear_spectrogram = np.dot(_inv_mel_basis, mel_spectrogram)
        if len(linear_spectrogram.shape) == 3:
            # for 3-dimension mel, the shape of
            # inverse linear spectrogram will be [num_freq, batch_size, n_frame]
            linear_spectrogram = np.transpose(linear_spectrogram, [1, 0, 2])
        return np.maximum(1e-10, linear_spectrogram)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def inv_mel_spectrogram(self, mel_spectrogram):
        S = self._mel_to_linear(self._db_to_amp(mel_spectrogram))
        return self._griffin_lim(S ** self.power)

    def preemphasize(self, x):
        if len(x.shape) == 1:  # [time_steps]
            return signal.lfilter([1, -self.pre_emphasize_value], [1], x)
        elif len(x.shape) == 2:  # [batch_size, time_steps]
            if x.shape[0] == 1:
                return np.expand_dims(
                    signal.lfilter([1, -self.pre_emphasize_value], [1], x[0]), axis=0)
            wav_list = list()
            for wav in x:
                wav_list.append(signal.lfilter([1, -self.pre_emphasize_value], [1], wav))
            return np.concatenate(wav_list, axis=0)
        else:
            raise Exception('Wave dimension error in pre-emphasis')

    def inv_preemphasize(self, x):
        if self.pre_emphasize_value is None:
            return x
        if len(x.shape) == 1:  # [time_steps]
            return signal.lfilter([1], [1, -self.pre_emphasize_value], x)
        elif len(x.shape) == 2:  # [batch_size, time_steps]
            if x.shape[0] == 1:
                return np.expand_dims(
                    signal.lfilter([1], [1, -self.pre_emphasize_value], x[0]), axis=0)
            wav_list = list()
            for wav in x:
                wav_list.append(signal.lfilter([1], [1, -self.pre_emphasize_value], wav))
            return np.concatenate(wav_list, axis=0)
        else:
            raise Exception('Wave dimension error in inverse pre-emphasis')

    def _normalize(self, S):
        return np.clip(self.max_abs_value * (
                (S - self.min_level_db) / (-self.min_level_db)),
                        0, self.max_abs_value)

    def _denormalize(self, S):
        return ((np.clip(S, 0, self.max_abs_value) * (-self.min_level_db)
                    / self.max_abs_value)
                + self.min_level_db)
