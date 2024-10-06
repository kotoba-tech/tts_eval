"""Pyannote speaker embedding model.
- pip install pyannote.audio
- feature dimension: 512
- source: https://huggingface.co/pyannote/embedding
"""
from typing import Optional, Union, Tuple
import torch
import numpy as np
from pyannote.audio import Model, Inference
from pyannote.audio.core.inference import fix_reproducibility, map_with_specifications


class PyannoteEmbedding:

    def __init__(self, device: Optional[Union[str, torch.device]] = None, *args, **kwargs):
        model = Model.from_pretrained("pyannote/embedding")
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.eval()
        self.inference = Inference(model, window="whole")

    def __call__(self, wav: np.ndarray, *args, **kwargs) -> np.ndarray:
        wav = torch.as_tensor(wav.reshape(1, -1), dtype=torch.float32).to(self.device)
        fix_reproducibility(self.inference.device)
        outputs = self.inference.infer(wav[None])

        def __first_sample(tmp: np.ndarray, **kwargs) -> np.ndarray:
            return tmp[0]

        return map_with_specifications(self.inference.model.specifications, __first_sample, outputs)
