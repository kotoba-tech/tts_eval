"""Speaker embedding obtained via speaker verification training.
- feature dimension: 256
- source: https://github.com/metavoiceio/metavoice-src
"""
import os
import subprocess
from os.path import join as p_join
from typing import Optional, Union

import librosa
import torch
import numpy as np
from librosa import feature

checkpoint_url = "https://github.com/kotoba-tech/tts_eval/releases/download/artifacts/meta_voice_speaker_encoder.pt"
model_weight = p_join(os.path.expanduser('~'), ".cache", "tts_eval", "meta_voice_speaker_encoder.pt")


def wget(url: str, output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    subprocess.run(["wget", url, "-O", output_file])
    if not os.path.exists(output_file):
        raise ValueError(f"failed to download {url}")


class MetaVoiceEmbedding(torch.nn.Module):

    mel_window_length: int = 25
    mel_window_step: int = 10
    mel_n_channels: int = 40
    sampling_rate: int = 16000
    partials_n_frames: int = 160
    model_hidden_size: int = 256
    model_embedding_size: int = 256
    model_num_layers: int = 3

    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 path_to_model_weight: Optional[str] = None,
                 *args,
                 **kwargs):
        super().__init__()
        path_to_model_weight = path_to_model_weight if path_to_model_weight else model_weight
        if not os.path.exists(path_to_model_weight):
            wget(checkpoint_url, path_to_model_weight)
        # Define the network
        self.lstm = torch.nn.LSTM(self.mel_n_channels, self.model_hidden_size, self.model_num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.model_hidden_size, self.model_embedding_size)
        self.relu = torch.nn.ReLU()
        # Load weight
        self.load_state_dict(torch.load(path_to_model_weight, map_location="cpu")["model_state"], strict=False)
        # Get the target device
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.eval()

    def compute_partial_slices(self, n_samples: int, rate: float, min_coverage: float) -> [np.ndarray, np.ndarray]:
        # Compute how many frames separate two partial utterances
        samples_per_frame = int((self.sampling_rate * self.mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = int(np.round((self.sampling_rate / rate) / samples_per_frame))
        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - self.partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + self.partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))
        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            return wav_slices[:-1], mel_slices[:-1]
        return wav_slices, mel_slices

    def __call__(self,
                 wav: np.ndarray,
                 sampling_rate: Optional[int] = None,
                 rate: float = 1.3,
                 min_coverage: float = 0.75) -> np.ndarray:
        if sampling_rate != self.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sampling_rate, target_sr=self.sampling_rate)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        wav_slices, mel_slices = self.compute_partial_slices(len(wav), rate, min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        # Wav -> Mel spectrogram
        frames = feature.melspectrogram(
            y=wav,
            sr=self.sampling_rate,
            n_fft=int(self.sampling_rate * self.mel_window_length / 1000),
            hop_length=int(self.sampling_rate * self.mel_window_step / 1000),
            n_mels=self.mel_n_channels,
        )
        mel = frames.astype(np.float32).T
        mel = np.array([mel[s] for s in mel_slices])
        # inference
        with torch.no_grad():
            mel = torch.from_numpy(mel).to(self.device)
            _, (hidden, _) = self.lstm(mel)
            embeds_raw = self.relu(self.linear(hidden[-1]))
            partial_embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        partial_embeds = partial_embeds.cpu().numpy()
        raw_embed = np.mean(partial_embeds, axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)
