from random import seed, shuffle
from itertools import chain
from datasets import load_dataset, concatenate_datasets, DatasetDict

repo_id = "kotoba-speech/tts_evaluation_v2"
source_repo_id = "kotoba-speech/youtube_crawl_audio_tscribed_audio_denoised_44_1kHz"
subsets_in = [
    ["voice_actors_academia", "jQ3MQyGqBBI"],
    ["joiito", "htAzZqnmlas"],
    ["weathernewstvch", "Ooeu7HKKBb8"],
    ["AppleTaka", "T_oVAW2L"],
    ["GenRonTV", "ZNyAcZsHBYY"],
    ["yuni_tarot", "V4O62I-8Sss"]
]
subsets_out = [
    ["nlp2980", "rxkXohKUthg"],
    ["citybunkyotv", "dRPIEJrtDS4"]
]
columns = ["key", "dataset_id", "audio", "text", "duration", "snr", "speaking_rate"]


def get_dataset(subset: str, key: str, sample_size: int = 25, n_additional_transcripts_per_sample: int = 4):
    dataset = load_dataset(source_repo_id, subset, split="train")
    dataset = dataset.filter(lambda x: key in x["key"])
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


dataset_dict = {}
for split_name, subsets in zip(["test_in", "test_out"], [subsets_in, subsets_out]):
    dataset_list = []
    text_transcript_list = []
    text_list = []
    for s, k in subsets:
        d, t_t, t = get_dataset(s, k)
        dataset_list.append(d)
        text_transcript_list.append(t_t)
        text_list.append(t)
    new_dataset = concatenate_datasets(dataset_list)
    new_text_transcript = list(chain(*text_transcript_list))
    new_dataset = new_dataset.add_column("transcript", new_text_transcript)
    new_dataset = new_dataset.add_column("text", new_text_transcript)
    new_text = list(chain(*text_list))
    seed(42)
    shuffle(new_text)
    dataset_list = [new_dataset]
    for s in range(0, len(new_text), len(new_text_transcript)):
        new_dataset = new_dataset.remove_columns("text")
        new_dataset = new_dataset.add_column("text", new_text[s: s + len(new_text_transcript)])
        dataset_list.append(new_dataset)
    new_dataset = concatenate_datasets(dataset_list)
    dataset_dict[split_name] = new_dataset
DatasetDict(dataset_dict).push_to_hub(repo_id, private=True)
