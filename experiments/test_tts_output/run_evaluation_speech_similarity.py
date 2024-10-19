import torch
from tqdm import tqdm

from datasets import load_dataset
from tts_eval import SpeakerEmbeddingSimilarity

columns_generated_speech = ['generated_audio_1', 'generated_audio_2', 'generated_audio_3', 'generated_audio_4']
column_audio = "audio"
column_text = "text"
dataset_id = "kotoba-speech/tts_evaluation_v1_with_audio"
speech_models = [
    "pyannote",
    "metavoice",
    "xlsr_2b",
    "hubert_xl",
    "w2v_bert"
]
for speech_model in speech_models:
    dataset = load_dataset(dataset_id, split="test")
    pipe = SpeakerEmbeddingSimilarity(
        model_id=speech_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa" if speech_model != "w2v_bert" else None
    )
    outputs = []
    for example in tqdm(dataset):
        outputs.append(pipe(
            audio_reference=example[column_audio]["array"],
            sampling_rate_reference=example[column_audio]["sampling_rate"],
            audio_target=[example[c]["array"] for c in columns_generated_speech],
            sampling_rate_target=[example[c]["sampling_rate"] for c in columns_generated_speech][0]
        ))
    for n, c in enumerate(columns_generated_speech):
        output = [o['cosine_similarity'][n] for o in outputs]
        dataset = dataset.add_column(f"{c}/speech_cos_sim.{speech_model}", output)
    dataset.push_to_hub(dataset_id, split="test", private=True)
