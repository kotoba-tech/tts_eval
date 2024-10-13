from random import seed, shuffle
from itertools import chain
from datasets import load_dataset, concatenate_datasets, DatasetDict

subsets = [
    "CVtY0yrLnyPqbcJb0NF8c6w",
    "CUOdWLM-qyApaHvW8SyviHg",
    "ChrisSasaki2020",
    "DISCOVER_WASEDA",
    "JINNAI-NETAJIN"
]
columns = [
    "key", "dataset_id", "audio", "text", "duration", "snr", "speaking_rate"
]


def get_dataset(subset: str, sample_size: int = 100, n_additional_transcripts_per_sample: int = 4):
    dataset = load_dataset("kotoba-speech/youtube_crawl_audio_tscribed_audio", subset, split="train")
    dataset = dataset.filter(lambda x: x["snr"] > 15)
    dataset = dataset.filter(lambda x: x["speaking_rate"] > 10)
    dataset = dataset.filter(lambda x: x["speaking_rate"] < 50)
    ind = list(range(len(dataset)))
    seed(42)
    shuffle(ind)
    text = dataset.select(ind[- sample_size * n_additional_transcripts_per_sample:])["text"]
    dataset = dataset.select(ind[:sample_size])
    text_transcript = dataset["text"]
    dataset = dataset.select_columns(columns)
    dataset = dataset.remove_columns("text")
    return dataset, text_transcript, text


dataset_list = []
text_transcript = []
text = []
for s in subsets:
    _dataset, _text_transcript, _text = get_dataset(s)
    dataset_list.append(_dataset)
    text_transcript.append(_text_transcript)
    text.append(_text)
new_dataset = concatenate_datasets(dataset_list)
new_text_transcript = list(chain(*text_transcript))
new_dataset = new_dataset.add_column("transcript", new_text_transcript)
new_dataset = new_dataset.add_column("text", new_text_transcript)
new_text = list(chain(*text))
seed(42)
shuffle(new_text)
dataset_list = [new_dataset]
for s in range(0, len(new_text), len(new_text_transcript)):
    new_dataset = new_dataset.remove_columns("text")
    new_dataset = new_dataset.add_column("text", new_text[s: s + len(new_text_transcript)])
    dataset_list.append(new_dataset)
new_dataset = concatenate_datasets(dataset_list)
DatasetDict({"test": new_dataset}).push_to_hub("kotoba-speech/tts_evaluation_v1", private=True)
