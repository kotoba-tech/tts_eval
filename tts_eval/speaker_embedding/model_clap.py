"""CLAP embedding.
- feature dimension: 512
- source: https://huggingface.co/laion/larger_clap_music_and_speech
"""
from typing import Optional, Union

import torch
import librosa
import numpy as np
from transformers import ClapModel, ClapProcessor


class CLAPEmbedding:
    def __init__(self,
                 ckpt: str = "laion/larger_clap_music_and_speech",
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None):
        model_config = {"torch_dtype": torch_dtype}
        if attn_implementation:
            model_config["model_kwargs"] = {"attn_implementation": attn_implementation}
        if device_map:
            self.model = ClapModel.from_pretrained(ckpt, device_map=device_map, **model_config)
            self.device = self.model.device
        else:
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = ClapModel.from_pretrained(ckpt, **model_config)
            self.model.to(self.device)
        self.model.eval()
        self.processor = ClapProcessor.from_pretrained(ckpt)

    def __call__(self, wav: np.ndarray, sampling_rate: Optional[int] = None) -> np.ndarray:
        if sampling_rate != self.processor.feature_extractor.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sampling_rate, target_sr=self.processor.feature_extractor.sampling_rate)
        inputs = self.processor(
            audios=wav, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
        return outputs.cpu().numpy()[0]


class CLAPGeneralEmbedding(CLAPEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None):
        super().__init__(
            ckpt="laion/larger_clap_general",
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )
