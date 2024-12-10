import ast
import json
import logging
import math
import os
import random
import h5py
from dataclasses import dataclass
import braceexpand
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from pathlib import Path
import wget
import tempfile
import copy
from contextlib import suppress

from clap_module.utils import get_tar_path_from_dataset_name, dataset_split
from clap_module.utils import load_p, load_class_label
from clap_module import tokenize as clip_tokenizer
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import BartTokenizer

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def tokenizer(text, tmodel="roberta", max_length=77):
    """Tokenizer for different models."""
    if tmodel == "roberta":
        result = roberta_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in result.items()}
    else:
        raise NotImplementedError(f"Tokenizer for {tmodel} not implemented")


# Initialized the audioset map
_AUDIOSET_MAP_PATH = os.path.join(Path(__file__).parent, "audioset_textmap.npy")
_AUDIOSET_MAP = np.load(_AUDIOSET_MAP_PATH, allow_pickle=True)


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)


def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_dataset_size(shards, sizefilepath_=None, is_local=True):
    if isinstance(shards, list):
        size_list = []
        for s in shards:
            size_list.append(
                get_dataset_size(s, sizefilepath_=sizefilepath_, is_local=is_local)[0]
            )
    else:
        if not is_local:
            for n in dataset_split.keys():
                if n in shards.split("/"):
                    break
            for s in dataset_split[n]:
                if s in shards.split("/"):
                    break
            sizefilepath_ = f"./json_files/{n}/{s}/sizes.json"
        shards_list = list(braceexpand.braceexpand(shards))
        dir_path = os.path.dirname(shards)
        if sizefilepath_ is not None:
            sizes = json.load(open(sizefilepath_, "r"))
            total_size = sum(
                [
                    int(sizes[os.path.basename(shard.replace(".tar -", ".tar"))])
                    for shard in shards_list
                ]
            )
        else:
            sizes_filename = os.path.join(dir_path, "sizes.json")
            len_filename = os.path.join(dir_path, "__len__")
            if os.path.exists(sizes_filename):
                sizes = json.load(open(sizes_filename, "r"))
                total_size = sum(
                    [int(sizes[os.path.basename(shard)]) for shard in shards_list]
                )
            elif os.path.exists(len_filename):
                total_size = ast.literal_eval(open(len_filename, "r").read())
            else:
                raise Exception(
                    f"Cannot find sizes file for dataset {shards}. Please specify the path to the file."
                )
    if isinstance(shards, list):
        return sum(size_list), len(shards)
    else:
        return total_size, len(shards_list)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


def sample_prop(sizefile, inputs, proportion, is_local=True):
    """
    Sample a proportion of the data.
    """
    file_path_dict = {
        os.path.split(inputs[i])[1]: os.path.split(inputs[i])[0]
        for i in range(len(inputs))
    }
    sampled_filepath_dict = {}
    sampled_size_dict = {}
    if not is_local:
        if os.path.exists("sizes.json"):
            os.remove("sizes.json")
        wget.download(sizefile, "sizes.json")
        sizefile = "sizes.json"
    with open(sizefile, "r", encoding="UTF-8") as f:
        load_dict = json.load(f)
    L = int(len(file_path_dict) * proportion)
    subkeys = random.sample(list(file_path_dict.keys()), L)
    for k in subkeys:
        sampled_size_dict[k] = load_dict[k]
        sampled_filepath_dict[k] = file_path_dict[k]
    return (
        sum(sampled_size_dict.values()),
        L,
        [os.path.join(v, k) for k, v in sampled_filepath_dict.items()],
        sampled_size_dict,
    )


def get_mel(audio_data, audio_cfg):
    # mel shape: (T, n_mels)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=audio_cfg['mel_bins'],
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    ).to(audio_data.device)
    mel = mel_tf(audio_data)
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)


def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
    """
    Calculate and add audio features to sample.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # Corner case where audio length is slightly larger than max_len
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    if len(ranges[1]) == 0:
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        ranges[2] = [0]
                    # Randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # Select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # Shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]

                    # Stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # Random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # Padding if too short
            if len(audio_data) < max_len:  # Do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample


def get_vocal_features(sample, vocal_data, max_len, data_truncating, data_filling, audio_cfg, require_grad=False):
    """
    Calculate and add vocal features to sample.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(vocal_data) > max_len:
            if data_truncating == "rand_trunc":
                # Random crop to max_len
                overflow = len(vocal_data) - max_len
                idx = np.random.randint(0, overflow + 1)
                vocal_data = vocal_data[idx: idx + max_len]
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented for vocal data"
                )
        else:  # Padding if too short
            if len(vocal_data) < max_len:
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(vocal_data))
                    vocal_data = vocal_data.repeat(n_repeat)
                    vocal_data = F.pad(
                        vocal_data,
                        (0, max_len - len(vocal_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    vocal_data = F.pad(
                        vocal_data,
                        (0, max_len - len(vocal_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(vocal_data))
                    vocal_data = vocal_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented for vocal data"
                    )
    sample["vocal_waveform"] = vocal_data

    return sample


def select_text(json_dict_raw, text_augment_selection):
    # For selecting augmented text from dataset
    if text_augment_selection is None or text_augment_selection == "none":
        texts = json_dict_raw["text"]
    elif text_augment_selection == "all":
        if "text_augment_all" in json_dict_raw.keys():
            texts = json_dict_raw["text_augment_all"]
        else:
            texts = json_dict_raw["text"]
    elif text_augment_selection == "augment_only":
        if "text_augment_all" in json_dict_raw.keys():
            if json_dict_raw["text_augment_t5"] is None:
                texts = json_dict_raw["text"]
            else:
                texts = json_dict_raw["text_augment_t5"]
        else:
            texts = json_dict_raw["text"]
    else:
        raise NotImplementedError(
            f"text_augment_selection {text_augment_selection} not implemented"
        )
    return texts


def preprocess_single(
        sample,
        audio_ext,
        text_ext,
        vocal_ext,
        max_len,
        audio_cfg,
        tmodel,
        class_index_dict,
        data_filling,
        data_truncating,
        text_augment_selection,
):
    """
    Preprocess a single sample for wdsdataloader.
    """
    # Process audio data
    audio_data, orig_sr = sample[audio_ext]
    audio_data = int16_to_float32_torch(float32_to_int16_torch(audio_data[0]))
    sample = get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg)
    del sample[audio_ext]

    # Process vocal data
    if vocal_ext in sample:
        vocal_data, vocal_sr = sample[vocal_ext]
        vocal_data = int16_to_float32_torch(float32_to_int16_torch(vocal_data[0]))
        sample = get_vocal_features(sample, vocal_data, max_len, data_truncating, data_filling, audio_cfg)
        del sample[vocal_ext]
        sample["vocal_orig_sr"] = vocal_sr
        sample["vocal_name"] = sample["__key__"].split("/")[-1] + "." + vocal_ext
    else:
        sample["vocal_waveform"] = torch.zeros(max_len)
        sample["vocal_orig_sr"] = orig_sr  # Default to audio's sample rate
        sample["vocal_name"] = None

    # Process text data
    json_dict_raw = sample[text_ext]
    texts = select_text(json_dict_raw, text_augment_selection)
    sample["full_text"] = texts

    if isinstance(texts, list) and isinstance(texts[0], str) and len(texts) > 1:
        texts = random.choice(texts)
    sample["raw_text"] = texts
    sample["text"] = tokenizer(texts, tmodel=tmodel)  # text shape: [num_token]
    if class_index_dict is not None:
        class_labels = np.zeros(len(class_index_dict))
        class_labels[np.in1d(list(class_index_dict.keys()), json_dict_raw["tag"])] = 1
        sample["class_label"] = torch.tensor(class_labels).float()

    del sample[text_ext]
    sample["audio_name"] = sample["__key__"].split("/")[-1] + "." + audio_ext
    sample["text_name"] = sample["__key__"].split("/")[-1] + "." + text_ext
    sample["audio_orig_sr"] = orig_sr
    return sample


def collate_fn_with_preprocess(batch,
                               audio_ext,
                               text_ext,
                               vocal_ext,
                               max_len,
                               audio_cfg,
                               args,
                               ):
    """
    Collate function for wdsdataloader.
    """
    class_index_dict = copy.deepcopy(args.class_index_dict)  # To avoid deadlock in multiprocessing
    data_filling = args.data_filling
    data_truncating = args.data_truncating
    text_augment_selection = args.text_augment_selection
    tmodel = args.tmodel

    data_preprocessed = []

    for sample in batch:
        data_preprocessed.append(
            preprocess_single(sample, audio_ext, text_ext, vocal_ext, max_len, audio_cfg, tmodel, class_index_dict,
                              data_filling, data_truncating, text_augment_selection))

    batch_dict = {}
    for k in data_preprocessed[0].keys():
        if isinstance(data_preprocessed[0][k], dict):  # Deal with tokenizer output
            batch_dict[k] = {}
            for kk in data_preprocessed[0][k].keys():
                tmp = []
                for i in range(len(data_preprocessed)):
                    tmp.append(data_preprocessed[i][k][kk])
                batch_dict[k][kk] = torch.vstack(tmp)
        elif isinstance(data_preprocessed[0][k], torch.Tensor):
            batch_dict[k] = torch.stack([sample[k] for sample in data_preprocessed])
        elif isinstance(data_preprocessed[0][k], np.ndarray):
            batch_dict[k] = torch.tensor(np.stack([sample[k] for sample in data_preprocessed]))
        else:
            batch_dict[k] = [sample[k] for sample in data_preprocessed]
    del data_preprocessed
    return batch_dict


def get_wds_dataset(
        args,
        model_cfg,
        is_train,
        audio_ext="flac",
        text_ext="json",
        vocal_ext="vocal.flac",  # Added for vocal data
        max_len=480000,
        proportion=1.0,
        sizefilepath_=None,
        is_local=None,
):
    """
    Get a dataset for wdsdataloader.
    """
    if is_local is None and (not args.remotedata is None):
        is_local = not args.remotedata

    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    if not sizefilepath_ is None:
        sizefilepath = sizefilepath_
    else:
        sizefilepath = os.path.join(os.path.dirname(input_shards[0]), "sizes.json")

    if proportion != 1.0:
        num_samples, num_shards, input_shards, _ = sample_prop(
            sizefilepath, input_shards, proportion, is_local=is_local
        )
    else:
        num_samples, num_shards = get_dataset_size(
            input_shards, sizefilepath_=sizefilepath_, is_local=is_local
        )

    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    "Currently, number of dataset samples must be specified for training dataset. "
                    "Please specify via `--train-num-samples` if no dataset length info present."
                )
        else:
            num_samples = (
                    args.val_num_samples or 0
            )  # eval will just exhaust the iterator if not specified

    pipeline = [wds.SimpleShardList(input_shards)]
    if is_train or args.parallel_eval:
        pipeline.extend(
            [
                wds.detshuffle(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                ),
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(
                    bufsize=_SAMPLE_SHUFFLE_SIZE,
                    initial=_SAMPLE_SHUFFLE_INITIAL,
                    rng=random.Random(args.seed),
                ),
            ]
        )
    else:
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
            ]
        )

    pipeline.append(
        wds.decode(wds.torch_audio),
    )

    pipeline.append(
        wds.batched(
            args.batch_size,
            partial=not (is_train or args.parallel_eval),
            collation_fn=partial(collate_fn_with_preprocess,
                                 audio_ext=audio_ext,
                                 text_ext=text_ext,
                                 vocal_ext=vocal_ext,
                                 max_len=max_len,
                                 audio_cfg=model_cfg['audio_cfg'],
                                 args=args,
                                 ),

        )
    )

    dataset = wds.DataPipeline(*pipeline)
    if is_train or args.parallel_eval:
        global_batch_size = args.batch_size * args.world_size
        num_batches = math.ceil(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = math.ceil(
            num_batches / num_workers
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(
            num_worker_batches
        )  # each worker is iterating over this
    else:
        num_batches = math.ceil(num_samples / args.batch_size)

    kwargs = {}
    if args.horovod:  # multi-node training
        kwargs["multiprocessing_context"] = "forkserver"

    if is_train:
        if args.prefetch_factor:
            prefetch_factor = args.prefetch_factor
        else:
            prefetch_factor = max(2, args.batch_size // args.workers)
    else:
        prefetch_factor = 2

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        **kwargs
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)


def get_dataset_fn(dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, model_cfg):
    data = {}

    args.class_index_dict = load_class_label(args.class_label_path)

    if args.datasetinfos is None:
        args.datasetinfos = ["train", "unbalanced_train", "balanced_train"]
    if args.dataset_type == "webdataset":
        args.train_data = get_tar_path_from_dataset_name(
            args.datasetnames,
            args.datasetinfos,
            islocal=not args.remotedata,
            proportion=args.dataset_proportion,
            dataset_path=args.datasetpath,
            full_dataset=args.full_train_dataset,
        )

        if args.full_train_dataset is None:
            args.full_train_dataset = []
        if args.exclude_eval_dataset is None:
            args.exclude_eval_dataset = []
        excluded_eval_datasets = args.full_train_dataset + args.exclude_eval_dataset

        val_dataset_names = [n for n in args.datasetnames if n not in excluded_eval_datasets] \
            if excluded_eval_datasets else args.datasetnames
        args.val_dataset_names = val_dataset_names
        args.val_data = get_tar_path_from_dataset_name(
            val_dataset_names,
            ["valid", "test", "eval"],
            islocal=not args.remotedata,
            proportion=1,
            dataset_path=args.datasetpath,
            full_dataset=None,
        )

    if args.train_data:
        data["train"] = get_dataset_fn(args.dataset_type)(
            args, model_cfg, is_train=True
        )

    if args.val_data:
        data["val"] = get_dataset_fn(args.dataset_type)(
            args, model_cfg, is_train=False
        )

    return data
