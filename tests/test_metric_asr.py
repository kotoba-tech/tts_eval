"""
wget https://huggingface.co/datasets/japanese-asr/ja_asr.jsut_basic5000/resolve/main/sample.flac -O sample_ja.flac
"""
from pprint import pprint
from tts_eval import ASRMetric

transcription = "水をマレーシアから買わなくてはならない"
audio = "sample_ja.flac"

pipe = ASRMetric(metrics="cer")
output = pipe([audio], transcript=transcription)
pprint(output)
