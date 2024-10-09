# TTS Eval
`tts_eval` is a python library for automatic evaluation of TTS outputs.


### Setup
```shell
pip install tts_eval
```

## Metrics
### ASR Metric
ASR metric evaluates fidelity of the generated speech by looking at the difference in the transcripts.
The reference transcript should be the prompt used as an input for the TTS generation and the transcript of the 
generated speech is predicted by ASR model.

***Python Usage:***

Get sample audio.
````shell
wget https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac -O sample_1.flac
wget https://huggingface.co/datasets/japanese-asr/en_asr.esb_eval/resolve/main/sample.wav -O sample_2.wav
````

Evaluate via python. 
```python
from tts_eval import ASRMetric
pipe = ASRMetric(
    model_id="kotoba-tech/kotoba-whisper-v2.0",  # ASR model to transcribe speech input
    metrics=["cer", "wer"]  # metrics
)
output = pipe(
    ["sample_1.flac", "sample_2.wav"],  # a list of audio to evaluate
    transcript="水をマレーシアから買わなくてはならない"  # reference transcript 
)
print(output)
{
    'cer': [15.789473684210526, 110.5263157894737],
    'wer': [100.0, 100.0]
}
```

### Speech Embedding Similarity
Speech embedding similarity evaluates voice cloning capability of TTS model.
It takes a reference speech used as speaker reference for generation, and compute similarity
between the reference speech and the generated speech based on speech embedding.

***Python Usage:***

Get sample audio.
````shell
wget https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac -O sample_1.flac
wget https://huggingface.co/datasets/japanese-asr/en_asr.esb_eval/resolve/main/sample.wav -O sample_2.wav
````

Evaluate via python. 
```python
from tts_eval import SpeakerEmbeddingSimilarity

pipe = SpeakerEmbeddingSimilarity(model_id="metavoice")
output = pipe(
    audio_target=["sample_1.flac", "sample_2.wav"],
    audio_reference="sample_1.flac"
)
print(output)
{
    'cosine_similarity': [1.0000001, 0.65718323]
}
```
Following speech embedding models are available:
- `metavoice`, `pyannote`, `clap`, `clap_general`, `w2v_bert`, `hubert_xl`, `hubert_large`, `hubert_base`, `wav2vec`, `xlsr_2b`, `xlsr_1b`, `xlsr_300m`