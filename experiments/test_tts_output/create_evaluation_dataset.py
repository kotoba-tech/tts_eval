from random import seed, shuffle
from itertools import chain
from datasets import load_dataset, concatenate_datasets, DatasetDict

repo_id = "kotoba-speech/tts_evaluation_v2"
source_repo_id = "kotoba-speech/youtube_crawl_audio_tscribed_audio_denoised_44_1kHz"
columns = ["key", "dataset_id", "audio", "text", "duration", "snr", "speaking_rate"]
sample_size = 25
random_seed = 42
n_additional_transcripts_per_sample = 4
# in-training channels
subsets_in = [
    ["voice_actors_academia", "jQ3MQyGqBBI"],
    ["joiito", "htAzZqnmlas"],
    ["weathernewstvch", "wkpE6pkQovs,xDkCJehyW-E"],
    ["AppleTaka", "T_oVAW2L"],
    ["GenRonTV", "ZNyAcZsHBYY"],
    ["yuni_tarot", "msLidLTtwpQ"]
]
# out-of-training channels
subsets_out = [
    ["nlp2980", "rxkXohKUthg"],
    ["citybunkyotv", "dRPIEJrtDS4,_bfDYZv7SJE,5lFZVin_6XY,VnIijzH-HMU"]
]


def get_index(dataset_size: int):
    ind = list(range(dataset_size))
    seed(random_seed)
    shuffle(ind)
    return ind


def get_dataset(subset: str, key: str):
    dataset = load_dataset(source_repo_id, subset, split="train")
    ind = get_index(len(dataset))
    text = dataset.select(ind[- sample_size * n_additional_transcripts_per_sample:])["text"]
    dataset = dataset.filter(lambda x: key in x["key"])
    dataset = dataset.filter(lambda x: x["snr"] > 15)
    dataset = dataset.filter(lambda x: x["speaking_rate"] > 10)
    dataset = dataset.filter(lambda x: x["speaking_rate"] < 50)
    ind = get_index(len(dataset))
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
        d, t, t_t = [], [], []
        for _k in k.split(","):
            _d, _t_t, _t = get_dataset(s, _k)
            d.append(_d)
            t_t += _t_t
            t += _t
        dataset_list.append(concatenate_datasets(d).select(range(sample_size)))
        text_transcript_list.append(t_t[:sample_size])
        text_list.append(t[:sample_size*n_additional_transcripts_per_sample])
    new_text_transcript = list(chain(*text_transcript_list))
    new_text = list(chain(*text_list))
    new_dataset = concatenate_datasets(dataset_list)
    new_dataset = new_dataset.add_column("transcript", new_text_transcript)
    new_dataset = new_dataset.add_column("text", new_text_transcript)
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
