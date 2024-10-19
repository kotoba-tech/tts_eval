import pandas as pd
from datasets import load_dataset, Dataset, Audio


data = load_dataset("kotoba-speech/tts_evaluation_v1_with_audio", split="test").to_pandas()
target_dataset_id = "kotoba-speech/tts_evaluation_v1_with_metrics"
column_reference = "audio"
column_transcript = "text"
column_transcript_original = "transcript"
column_generations = ['generated_audio_1', 'generated_audio_2', 'generated_audio_3', 'generated_audio_4']
metrics = [
    "cer",
    "speech_cos_sim.pyannote",
    "speech_cos_sim.metavoice",
    "speech_cos_sim.xlsr_2b",
    "speech_cos_sim.hubert_xl",
    "speech_cos_sim.w2v_bert"
]


def format_df(metric, text_transcript: bool = False):
    list_df = []
    for c in column_generations:
        tmp_df = data[[column_transcript, column_transcript_original, column_reference, c, f"{c}/{metric}"]]
        tmp_df.columns = ["transcript", "transcript_original", "reference", "generation", metric]
        if text_transcript:
            tmp_df = tmp_df[[x == y for x, y in zip(tmp_df["transcript"], tmp_df["transcript_original"])]]
        list_df.append(tmp_df[["transcript", "reference", "generation", metric]])
    df = pd.concat(list_df).sort_values(by=metric, ascending=True)
    tmp_ds = Dataset.from_pandas(df)
    tmp_ds = tmp_ds.cast_column('reference', Audio())
    tmp_ds = tmp_ds.cast_column('generation', Audio())
    tmp_ds = tmp_ds.remove_columns("__index_level_0__")
    return tmp_ds


for m in metrics:
    ds = format_df(m)
    ds.push_to_hub(target_dataset_id, config_name=f"random_input.{m}", split="test", private=True)
for m in metrics:
    ds = format_df(m, text_transcript=True)
    ds.push_to_hub(target_dataset_id, config_name=f"transcript_input.{m}", split="test", private=True)

