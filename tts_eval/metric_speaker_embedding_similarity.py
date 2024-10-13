"""Embedding based metric to measure the adhesiveness of the generated audio to its reference speaker."""

import logging
from typing import Optional, Union, List, Dict

import soundfile as sf
import torch
import numpy as np

from numpy import dot
from numpy.linalg import norm
from .speaker_embedding import speaker_embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray):
    return dot(a, b) / (norm(a) * norm(b))


def l2_distance(a: np.ndarray, b: np.ndarray):
    return norm(a - b)


class SpeakerEmbeddingSimilarity:

    def __init__(self,
                 model_id: str = "pyannote",
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None):
        logging.info("setup pipeline")
        self.model = speaker_embeddings[model_id](
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )

    def __call__(self,
                 audio_target: List[Union[np.ndarray, str]],
                 audio_reference: Union[np.ndarray, bytes, str],
                 sampling_rate_target: Optional[int] = None,
                 sampling_rate_reference: Optional[int] = None,
                 metric: str = "cosine_similarity") -> Dict[str, List[float]]:
        if type(audio_reference) is str:
            a, sr = sf.read(audio_reference)
        else:
            a, sr = audio_reference, sampling_rate_reference
        vector_reference = self.model(wav=a, sampling_rate=sr)
        scores = []
        for i in audio_target:
            if type(i) is str:
                a, sr = sf.read(i)
            else:
                a, sr = i, sampling_rate_target
            vector_target = self.model(wav=a, sampling_rate=sr)
            if metric == "cosine_similarity":
                scores.append(cosine_similarity(vector_reference, vector_target))
            elif metric == "negative_l2_distance":
                scores.append(- l2_distance(vector_reference, vector_target))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return {metric: scores}
