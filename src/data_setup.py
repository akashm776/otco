import os
from dataclasses import dataclass
from functools import partial

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.flickr8k_dataset import (
    Flickr8kAllCaptionsDataset,
    Flickr8kCanonicalCaptionDataset,
    Flickr8kUniqueImageDataset,
    caption_collate_batch,
    seed_worker as local_seed_worker,
)
from data.hf_flickr8k_dataset import (
    HFFlickr8kAllCaptionsDataset,
    HFFlickr8kCanonicalCaptionDataset,
    HFFlickr8kUniqueImageDataset,
    load_hf_flickr8k_splits,
    seed_worker as hf_seed_worker,
)
from data.hf_flickr30k_dataset import (
    HFFlickr30kAllCaptionsDataset,
    HFFlickr30kCanonicalCaptionDataset,
    HFFlickr30kUniqueImageDataset,
    load_hf_flickr30k_splits,
)
from data.hf_cub200_dataset import (
    HFCUB200AllCaptionsDataset,
    HFCUB200CanonicalCaptionDataset,
    HFCUB200UniqueImageDataset,
    get_cub200_class_labels,
    load_hf_cub200_splits,
)
from data.stratified_sampler import StratifiedClassSampler


@dataclass
class DataBundle:
    tokenizer: object
    train_dataset: object
    val_dataset_all: object
    val_dataset_canonical: object
    train_loader: object
    val_loader_all: object
    val_loader_canonical: object
    stats: dict
    stratified_sampler: object = None


def _load_local_flickr8k_from_disk(root_dir, train_split):
    captions = []
    caption_list = []
    images = []
    curr = None
    icurr = None
    flag = True

    with open(f"{root_dir}/data/datasets/Flickr8k/captions.txt", "r", encoding="utf-8") as f:
        for line in f:
            if flag:
                flag = False
                continue
            line = line.split("\n")[0]
            image, caption = line.split(",", 1)

            if icurr is None:
                images.append(f"{root_dir}/data/datasets/Flickr8k/Images/{image}")
                icurr = image
            elif icurr != image:
                images.append(f"{root_dir}/data/datasets/Flickr8k/Images/{image}")
                icurr = image

            if len(caption_list) == 0:
                curr = image
                caption_list.append(caption)
            elif curr == image:
                caption_list.append(caption)
            else:
                captions.append(caption_list)
                caption_list = [caption]
                curr = image

        captions.append(caption_list)

    num_images = len(images)
    train_size = int(train_split * num_images)
    train_images = images[:train_size]
    train_captions = captions[:train_size]
    val_images = images[train_size:]
    val_captions = captions[train_size:]
    return train_images, train_captions, val_images, val_captions


def _build_local_datasets(root_dir, config):
    train_images, train_captions, val_images, val_captions = _load_local_flickr8k_from_disk(
        root_dir=root_dir,
        train_split=config["train_split"],
    )
    train_dataset = Flickr8kUniqueImageDataset(train_images, train_captions, is_train=True)
    val_dataset_all = Flickr8kAllCaptionsDataset(val_images, val_captions)
    val_dataset_canonical = Flickr8kCanonicalCaptionDataset(val_images, val_captions)
    return train_dataset, val_dataset_all, val_dataset_canonical, local_seed_worker, {
        "dataset_backend": "local_flickr8k",
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
    }


def _build_hf_datasets(config, dataset_name, hf_train_split, hf_val_split):
    train_grouped, val_grouped, split_info = load_hf_flickr8k_splits(
        dataset_name=dataset_name,
        train_hf_split=hf_train_split,
        val_hf_split=hf_val_split,
        train_split=config["train_split"],
        seed=config["seed"],
    )
    train_dataset = HFFlickr8kUniqueImageDataset(train_grouped, is_train=True)
    val_dataset_all = HFFlickr8kAllCaptionsDataset(val_grouped)
    val_dataset_canonical = HFFlickr8kCanonicalCaptionDataset(val_grouped)

    stats = {
        "dataset_backend": "hf_flickr8k",
        "hf_dataset_name": dataset_name,
        "num_train_images": len(train_grouped.groups),
        "num_val_images": len(val_grouped.groups),
    }
    stats.update(split_info)
    return train_dataset, val_dataset_all, val_dataset_canonical, hf_seed_worker, stats


def _build_hf_flickr30k_datasets(config, dataset_name, hf_train_split, hf_val_split):
    train_grouped, val_grouped, split_info = load_hf_flickr30k_splits(
        dataset_name=dataset_name,
        train_hf_split=hf_train_split,
        val_hf_split=hf_val_split,
        train_split=config["train_split"],
        seed=config["seed"],
    )
    train_dataset = HFFlickr30kUniqueImageDataset(train_grouped, is_train=True)
    val_dataset_all = HFFlickr30kAllCaptionsDataset(val_grouped)
    val_dataset_canonical = HFFlickr30kCanonicalCaptionDataset(val_grouped)

    stats = {
        "dataset_backend": "hf_flickr30k",
        "hf_dataset_name": dataset_name,
        "num_train_images": len(train_grouped.groups),
        "num_val_images": len(val_grouped.groups),
    }
    stats.update(split_info)
    return train_dataset, val_dataset_all, val_dataset_canonical, hf_seed_worker, stats


def _build_hf_cub200_datasets(config, dataset_name, hf_train_split, hf_val_split):
    train_grouped, val_grouped, split_info = load_hf_cub200_splits(
        dataset_name=dataset_name,
        train_hf_split=hf_train_split,
        val_hf_split=hf_val_split,
    )
    train_dataset = HFCUB200UniqueImageDataset(train_grouped, is_train=True)
    val_dataset_all = HFCUB200AllCaptionsDataset(val_grouped)
    val_dataset_canonical = HFCUB200CanonicalCaptionDataset(val_grouped)
    class_labels = get_cub200_class_labels(train_grouped)

    stats = {
        "dataset_backend": "hf_cub200",
        "hf_dataset_name": dataset_name,
        "num_train_images": len(train_grouped.groups),
        "num_val_images": len(val_grouped.groups),
    }
    stats.update(split_info)
    return train_dataset, val_dataset_all, val_dataset_canonical, hf_seed_worker, stats, class_labels


def build_data_bundle(
    config,
    root_dir=None,
    dataset_cfg=None,
):
    if dataset_cfg is None:
        dataset_cfg = {}
    dataset_backend = dataset_cfg.get("backend", "local_flickr8k")
    local_cfg = dataset_cfg.get("local", {}) or {}
    hf_cfg = dataset_cfg.get("hf", {}) or {}

    if root_dir is None:
        root_dir = local_cfg.get("root_dir") or os.path.abspath(os.curdir)

    tokenizer = AutoTokenizer.from_pretrained(config["model_text"])
    collate_fn = partial(caption_collate_batch, tokenizer=tokenizer)

    class_labels = None
    if dataset_backend == "local_flickr8k":
        train_dataset, val_dataset_all, val_dataset_canonical, seed_fn, stats = _build_local_datasets(
            root_dir=root_dir,
            config=config,
        )
    elif dataset_backend == "hf_flickr8k":
        train_dataset, val_dataset_all, val_dataset_canonical, seed_fn, stats = _build_hf_datasets(
            config=config,
            dataset_name=hf_cfg.get("dataset_name", "nlphuji/flickr8k"),
            hf_train_split=hf_cfg.get("train_split", "train"),
            hf_val_split=hf_cfg.get("val_split"),
        )
    elif dataset_backend == "hf_flickr30k":
        train_dataset, val_dataset_all, val_dataset_canonical, seed_fn, stats = _build_hf_flickr30k_datasets(
            config=config,
            dataset_name=hf_cfg.get("dataset_name", "nlphuji/flickr30k"),
            hf_train_split=hf_cfg.get("train_split", "test"),
            hf_val_split=hf_cfg.get("val_split"),
        )
    elif dataset_backend == "hf_cub200":
        train_dataset, val_dataset_all, val_dataset_canonical, seed_fn, stats, class_labels = _build_hf_cub200_datasets(
            config=config,
            dataset_name=hf_cfg.get("dataset_name", "alkzar90/CC6204-Hackaton-Cub-Dataset"),
            hf_train_split=hf_cfg.get("train_split", "train"),
            hf_val_split=hf_cfg.get("val_split", "test"),
        )
    else:
        raise ValueError(f"Unknown dataset backend: {dataset_backend}")

    use_stratified = config.get("stratified_batching", False) and class_labels is not None
    stratified_sampler = None

    if use_stratified:
        classes_per_batch = config.get("classes_per_batch", config["batch_size"] // 4)
        images_per_class = config["batch_size"] // classes_per_batch
        stratified_sampler = StratifiedClassSampler(
            class_labels=class_labels,
            classes_per_batch=classes_per_batch,
            images_per_class=images_per_class,
            seed=config["seed"],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=stratified_sampler,
            collate_fn=collate_fn,
            num_workers=config["num_workers"],
            worker_init_fn=seed_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config["num_workers"],
            worker_init_fn=seed_fn,
            drop_last=config["drop_last"],
        )
    val_loader_all = DataLoader(
        val_dataset_all,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        worker_init_fn=seed_fn,
    )
    val_loader_canonical = DataLoader(
        val_dataset_canonical,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["num_workers"],
        worker_init_fn=seed_fn,
    )

    if use_stratified:
        steps_per_epoch = len(stratified_sampler)
    else:
        steps_per_epoch = len(train_dataset) // config["batch_size"]
        if not config["drop_last"] and len(train_dataset) % config["batch_size"] != 0:
            steps_per_epoch += 1
    total_steps = config["num_epochs"] * steps_per_epoch

    stats.update(
        {
            "num_train_samples": len(train_dataset),
            "num_val_all_samples": len(val_dataset_all),
            "num_val_canonical_samples": len(val_dataset_canonical),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        }
    )

    return DataBundle(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset_all=val_dataset_all,
        val_dataset_canonical=val_dataset_canonical,
        train_loader=train_loader,
        val_loader_all=val_loader_all,
        val_loader_canonical=val_loader_canonical,
        stats=stats,
        stratified_sampler=stratified_sampler,
    )
