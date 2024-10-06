from .model_meta_voice import MetaVoiceEmbedding
from .model_pyannote_embedding import PyannoteEmbedding
from .model_clap import CLAPEmbedding, CLAPGeneralEmbedding
from .model_hf import (
    W2VBERTEmbedding,
    HuBERTXLEmbedding,
    HuBERTLargeEmbedding,
    HuBERTBaseEmbedding,
    Wav2VecEmbedding,
    XLSR2BEmbedding,
    XLSR1BEmbedding,
    XLSR300MEmbedding
)

speaker_embeddings = {
    "metavoice": MetaVoiceEmbedding,
    "pyannote": PyannoteEmbedding,
    "clap": CLAPEmbedding,
    "clap_general": CLAPGeneralEmbedding,
    "w2v_bert": W2VBERTEmbedding,
    "hubert_xl": HuBERTXLEmbedding,
    "hubert_large": HuBERTLargeEmbedding,
    "hubert_base": HuBERTBaseEmbedding,
    "wav2vec": Wav2VecEmbedding,
    "xlsr_2b": XLSR2BEmbedding,
    "xlsr_1b": XLSR1BEmbedding,
    "xlsr_300m": XLSR300MEmbedding
}
