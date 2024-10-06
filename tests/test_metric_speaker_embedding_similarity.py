"""
wget https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac -O sample_ja.flac
wget https://huggingface.co/datasets/japanese-asr/en_asr.esb_eval/resolve/main/sample.wav -O sample_en.wav
"""
from pprint import pprint
from tts_eval import SpeakerEmbeddingSimilarity

audio = ["sample_ja.flac", "sample_en.wav"]

pipe = SpeakerEmbeddingSimilarity()
output = pipe(audio, audio[0])
pprint(output)
