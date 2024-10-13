from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets, DatasetDict
from tts_eval import ASRMetric, SpeakerEmbeddingSimilarity

columns_generated_speech = ['generated_audio_1', 'generated_audio_2', 'generated_audio_3', 'generated_audio_4']
column_text = "text"
dataset_id = "kotoba-speech/tts_evaluation_v1_with_audio"
asr_model = "kotoba-tech/kotoba-whisper-v2.0"
pipe = ASRMetric(model_id=asr_model, metrics=["cer", "wer"])
dataset = load_dataset(dataset_id, split="test")
outputs = []
for example in tqdm(dataset):
    outputs.append(pipe(
        transcript=example[column_text],
        audio=[example[c] for c in columns_generated_speech]
    ))

for n, c in enumerate(columns_generated_speech):
    output = [o['cer'][n] for o in outputs]
    dataset = dataset.add_column(f"{c}/cer", output)
dataset.push_to_hub(dataset_id, split="test", private=True)
