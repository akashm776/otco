"""Deterministic CUB-200 datasets for the isolated CLIP training path."""

from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset

from data.hf_cub200_dataset import (
    get_cub200_class_labels,
    load_hf_cub200_splits,
    seed_worker,
)
from data.hf_flickr8k_dataset import _decode_image
from src.clip_geometry_diagnostic_v2 import species_name_from_image_key


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def load_diagnostic_holdout(path, original_count):
    """Load the exact V1/V2 source indices and validate their integrity."""
    resolved = resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        indices = json.load(handle)
    if not isinstance(indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ):
        raise TypeError("Diagnostic holdout must be a JSON list of integer indices")
    if len(indices) != len(set(indices)):
        raise ValueError("Diagnostic holdout contains duplicate indices")
    if any(index < 0 or index >= original_count for index in indices):
        raise ValueError("Diagnostic holdout contains an out-of-range index")
    return indices


def exclude_diagnostic_indices(original_count, holdout_indices):
    """Return the complement and prove it is disjoint from the holdout."""
    holdout = set(holdout_indices)
    training = [index for index in range(original_count) if index not in holdout]
    if set(training) & holdout:
        raise AssertionError("Training and diagnostic holdout indices overlap")
    if len(training) + len(holdout) != original_count:
        raise AssertionError("Training/holdout partition does not cover the source")
    return training


def build_species_metadata(grouped_split):
    """Create stable numeric class IDs and one prompt-ready name per species."""
    labels = get_cub200_class_labels(grouped_split)
    ordered_labels = list(dict.fromkeys(labels))
    label_to_id = {label: index for index, label in enumerate(ordered_labels)}
    numeric_ids = [label_to_id[label] for label in labels]
    names = {}
    for group, species_id in zip(grouped_split.groups, numeric_ids):
        names.setdefault(species_id, species_name_from_image_key(group.image_key))
    return numeric_ids, names


class CLIPCUBPairDataset(Dataset):
    """Raw PIL image-caption pairs processed only by the checkpoint processor."""

    def __init__(
        self,
        grouped_split,
        species_ids,
        *,
        source_indices=None,
        caption_view="epoch_random",
        seed=42,
    ):
        self.grouped_split = grouped_split
        self.species_ids = list(species_ids)
        self.source_indices = list(
            range(len(grouped_split.groups))
            if source_indices is None
            else source_indices
        )
        self.caption_view = caption_view
        self.seed = seed
        self.epoch = 0
        if caption_view not in {"epoch_random", "canonical_first_caption"}:
            raise ValueError(f"Unsupported caption view: {caption_view}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.source_indices)

    def __getitem__(self, index):
        source_index = self.source_indices[index]
        group = self.grouped_split.groups[source_index]
        row = self.grouped_split.hf_split[group.row_index]
        image = _decode_image(row[self.grouped_split.schema.image_column])
        if self.caption_view == "canonical_first_caption":
            caption_index = 0
        else:
            rng = random.Random(
                self.seed + self.epoch * 1_000_003 + source_index
            )
            caption_index = rng.randrange(len(group.captions))
        return {
            "image": image,
            "caption": group.captions[caption_index],
            "image_id": index,
            "source_index": source_index,
            "caption_index": caption_index,
            "species_id": self.species_ids[source_index],
            "image_key": group.image_key,
        }


class CLIPCUBAllCaptionsDataset(Dataset):
    """All validation captions with local image IDs for retrieval evaluation."""

    def __init__(self, grouped_split, species_ids):
        self.grouped_split = grouped_split
        self.species_ids = list(species_ids)
        self.entries = [
            (image_index, caption_index)
            for image_index, group in enumerate(grouped_split.groups)
            for caption_index in range(len(group.captions))
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        image_index, caption_index = self.entries[index]
        group = self.grouped_split.groups[image_index]
        return {
            "image": None,
            "caption": group.captions[caption_index],
            "image_id": image_index,
            "source_index": image_index,
            "caption_index": caption_index,
            "species_id": self.species_ids[image_index],
            "image_key": group.image_key,
        }


class CLIPProcessorCollator:
    """Apply the matching CLIP processor to both modalities in one place."""

    def __init__(self, processor, *, include_images=True):
        self.processor = processor
        self.include_images = include_images

    def __call__(self, records):
        arguments = {
            "text": [record["caption"] for record in records],
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.include_images:
            arguments["images"] = [record["image"] for record in records]
        processed = self.processor(**arguments)
        batch = {
            "input_ids": processed["input_ids"],
            "species_ids": torch.tensor(
                [record["species_id"] for record in records], dtype=torch.long
            ),
            "image_ids": torch.tensor(
                [record["image_id"] for record in records], dtype=torch.long
            ),
            "source_indices": torch.tensor(
                [record["source_index"] for record in records], dtype=torch.long
            ),
            "captions": [record["caption"] for record in records],
            "image_keys": [record["image_key"] for record in records],
        }
        if "attention_mask" in processed:
            batch["attention_mask"] = processed["attention_mask"]
        if "pixel_values" in processed:
            batch["pixel_values"] = processed["pixel_values"]
        return batch


def make_loader(
    dataset,
    processor,
    *,
    batch_size,
    shuffle,
    drop_last,
    num_workers,
    seed,
    pin_memory,
    include_images=True,
):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=CLIPProcessorCollator(
            processor, include_images=include_images
        ),
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=pin_memory,
        # Workers are recreated each epoch so set_epoch() reaches their dataset
        # copies and the deterministic epoch-specific caption choice is real.
        persistent_workers=False,
    )


@dataclass
class CLIPTrainingData:
    train_dataset: CLIPCUBPairDataset
    validation_canonical_dataset: CLIPCUBPairDataset
    validation_all_captions_dataset: CLIPCUBAllCaptionsDataset
    train_loader: DataLoader
    validation_canonical_loader: DataLoader
    validation_all_captions_loader: DataLoader
    species_names: dict
    exclusion_report: dict
    split_info: dict


def build_clip_training_data(config, processor, *, pin_memory):
    dataset_config = config["dataset"]
    train_grouped, validation_grouped, split_info = load_hf_cub200_splits(
        dataset_name=dataset_config["dataset_name"],
        train_hf_split=dataset_config["train_split"],
        val_hf_split=dataset_config["validation_split"],
    )
    train_species, train_species_names = build_species_metadata(train_grouped)
    validation_species, validation_species_names = build_species_metadata(
        validation_grouped
    )
    if set(train_species_names.values()) != set(validation_species_names.values()):
        raise AssertionError("Training and validation CUB species sets differ")

    original_count = len(train_grouped.groups)
    holdout = load_diagnostic_holdout(
        dataset_config["diagnostic_holdout_indices"], original_count
    )
    training_indices = exclude_diagnostic_indices(original_count, holdout)
    train_dataset = CLIPCUBPairDataset(
        train_grouped,
        train_species,
        source_indices=training_indices,
        caption_view=dataset_config["caption_sampling"],
        seed=config["training"]["seed"],
    )
    canonical_dataset = CLIPCUBPairDataset(
        validation_grouped,
        validation_species,
        caption_view="canonical_first_caption",
        seed=config["training"]["seed"],
    )
    all_captions_dataset = CLIPCUBAllCaptionsDataset(
        validation_grouped, validation_species
    )
    loader_options = {
        "processor": processor,
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["runtime"]["num_workers"],
        "seed": config["training"]["seed"],
        "pin_memory": pin_memory,
    }
    train_loader = make_loader(
        train_dataset, shuffle=True, drop_last=True, **loader_options
    )
    canonical_loader = make_loader(
        canonical_dataset, shuffle=False, drop_last=False, **loader_options
    )
    all_captions_loader = make_loader(
        all_captions_dataset,
        shuffle=False,
        drop_last=False,
        include_images=False,
        **loader_options,
    )
    return CLIPTrainingData(
        train_dataset=train_dataset,
        validation_canonical_dataset=canonical_dataset,
        validation_all_captions_dataset=all_captions_dataset,
        train_loader=train_loader,
        validation_canonical_loader=canonical_loader,
        validation_all_captions_loader=all_captions_loader,
        species_names=validation_species_names,
        exclusion_report={
            "original_training_count": original_count,
            "diagnostic_exclusion_count": len(holdout),
            "final_training_count": len(training_indices),
            "intersection_count": len(set(training_indices) & set(holdout)),
            "holdout_path": str(
                resolve_project_path(dataset_config["diagnostic_holdout_indices"])
            ),
        },
        split_info=split_info,
    )
