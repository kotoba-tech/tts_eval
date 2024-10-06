"""Speaker embedding model from HuggingFace Transformers"""
from typing import Optional, Union

import torch
import librosa
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor


############
# W2V BERT #
############
class W2VBERTEmbedding:
    def __init__(self,
                 ckpt: str = "facebook/w2v-bert-2.0",
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        self.processor = AutoFeatureExtractor.from_pretrained(ckpt)
        model_config = {"torch_dtype": torch_dtype}
        if attn_implementation:
            model_config["model_kwargs"] = {"attn_implementation": attn_implementation}
        if device_map:
            self.model = AutoModel.from_pretrained(ckpt, device_map=device_map, **model_config)
            self.device = self.model.device
        else:
            self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = AutoModel.from_pretrained(ckpt, **model_config)
            self.model.to(self.device)
        self.model.eval()
        self.mean_pool = mean_pool

    def __call__(self, wav: np.ndarray, sampling_rate: Optional[int] = None) -> np.ndarray:
        # audio file is decoded on the fly
        if sampling_rate != self.processor.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sampling_rate, target_sr=self.processor.sampling_rate)
        inputs = self.processor(wav, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        if self.mean_pool:
            return outputs.last_hidden_state.mean(1).cpu().numpy()[0]
        return outputs.last_hidden_state.cpu().numpy()[0]


##########
# HuBERT #
##########
class HuBERTXLEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/hubert-xlarge-ll60k",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )


class HuBERTLargeEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/hubert-large-ll60k",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )


class HuBERTBaseEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/hubert-base-ls960",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )


###########
# wav2vec #
###########
class Wav2VecEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/wav2vec2-large-xlsr-53",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )


#########
# XLS-R #
#########
class XLSR2BEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/wav2vec2-xls-r-2b",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )


class XLSR1BEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/wav2vec2-xls-r-1b",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )


class XLSR300MEmbedding(W2VBERTEmbedding):
    def __init__(self,
                 device: Optional[Union[str, torch.device]] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 attn_implementation: Optional[str] = None,
                 mean_pool: bool = True):
        super().__init__(
            "facebook/wav2vec2-xls-r-300m",
            mean_pool=mean_pool,
            device=device,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation
        )
