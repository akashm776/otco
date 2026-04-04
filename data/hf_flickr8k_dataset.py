import io
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


IMAGE_COLUMN_CANDIDATES = ["image", "img", "photo"]
CAPTION_COLUMN_CANDIDATES = ["caption", "captions", "text", "sentence", "sentences", "description"]
IMAGE_ID_COLUMN_CANDIDATES = [
    "image_id",
    "img_id",
    "id",
    "filename",
    "file_name",
    "image_path",
    "filepath",
    "path",
]
CAPTION_DICT_KEYS = ["raw", "caption", "text", "sentence"]


@dataclass
class HFColumnSchema:
    image_column: str
    caption_column: str
    image_id_column: str | None


@dataclass
class HFGroupedImage:
    row_index: int
    image_key: str
    captions: list[str]


@dataclass
class HFGroupedSplit:
    split_name: str
    hf_split: object
    schema: HFColumnSchema
    groups: list[HFGroupedImage]


def _is_image_like(value):
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, dict) and ("path" in value or "bytes" in value):
        return True
    return False


def _extract_caption_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(v, str) for v in value):
            return value
        if all(isinstance(v, dict) for v in value):
            captions = []
            for item in value:
                for key in CAPTION_DICT_KEYS:
                    if key in item and isinstance(item[key], str):
                        captions.append(item[key])
                        break
            return captions
    if isinstance(value, dict):
        for key in CAPTION_DICT_KEYS:
            if key in value and isinstance(value[key], str):
                return [value[key]]
    raise ValueError(f"Unsupported caption type: {type(value)}")


def _to_image_key(example, image_value, schema, row_index, captions):
    if schema.image_id_column and schema.image_id_column in example:
        value = example[schema.image_id_column]
        if isinstance(value, (str, int)):
            return str(value)

    for key in ("image_path", "path", "filename", "file_name"):
        if key in example and isinstance(example[key], str):
            return example[key]

    if isinstance(image_value, dict) and isinstance(image_value.get("path"), str):
        return image_value["path"]

    if isinstance(image_value, Image.Image):
        filename = getattr(image_value, "filename", None)
        if isinstance(filename, str) and filename:
            return filename

    # If each row already contains all captions for that image, row index is stable.
    if len(captions) > 1:
        return f"row:{row_index}"

    raise ValueError(
        "Could not infer stable image key for grouping captions. "
        "Provide a dataset with image id/path columns."
    )


def _detect_schema(hf_split):
    if len(hf_split) == 0:
        raise ValueError("Cannot infer schema from an empty split.")
    example = hf_split[0]
    columns = list(example.keys())

    image_column = None
    for name in IMAGE_COLUMN_CANDIDATES:
        if name in example and _is_image_like(example[name]):
            image_column = name
            break
    if image_column is None:
        for name in columns:
            if _is_image_like(example[name]):
                image_column = name
                break
    if image_column is None:
        raise ValueError(f"Could not infer image column from keys: {columns}")

    caption_column = None
    for name in CAPTION_COLUMN_CANDIDATES:
        if name in example:
            try:
                _extract_caption_strings(example[name])
                caption_column = name
                break
            except Exception:
                pass
    if caption_column is None:
        for name in columns:
            try:
                _extract_caption_strings(example[name])
                caption_column = name
                break
            except Exception:
                pass
    if caption_column is None:
        raise ValueError(f"Could not infer caption column from keys: {columns}")

    image_id_column = None
    for name in IMAGE_ID_COLUMN_CANDIDATES:
        if name in example:
            image_id_column = name
            break

    return HFColumnSchema(
        image_column=image_column,
        caption_column=caption_column,
        image_id_column=image_id_column,
    )


def group_hf_split_by_image(hf_split, split_name):
    schema = _detect_schema(hf_split)
    grouped = {}
    ordered_keys = []

    for row_index in range(len(hf_split)):
        example = hf_split[row_index]
        image_value = example[schema.image_column]
        captions = _extract_caption_strings(example[schema.caption_column])
        if not captions:
            continue
        image_key = _to_image_key(example, image_value, schema, row_index, captions)

        if image_key not in grouped:
            grouped[image_key] = HFGroupedImage(
                row_index=row_index,
                image_key=image_key,
                captions=[],
            )
            ordered_keys.append(image_key)

        grouped[image_key].captions.extend(captions)

    groups = [grouped[key] for key in ordered_keys]
    return HFGroupedSplit(split_name=split_name, hf_split=hf_split, schema=schema, groups=groups)


def load_hf_flickr8k_splits(
    dataset_name="nlphuji/flickr8k",
    train_hf_split="train",
    val_hf_split=None,
    train_split=0.9,
    seed=42,
):
    if load_dataset is None:
        raise ImportError(
            "datasets library is required for Hugging Face loading. "
            "Install with: pip install datasets"
        )

    ds = load_dataset(dataset_name)
    available_splits = list(ds.keys())

    if train_hf_split not in ds:
        raise ValueError(f"Train split '{train_hf_split}' not found in dataset. Available: {available_splits}")

    train_grouped = group_hf_split_by_image(ds[train_hf_split], split_name=train_hf_split)

    chosen_val_split = val_hf_split
    if chosen_val_split is None:
        for candidate in ("validation", "val", "test"):
            if candidate in ds and candidate != train_hf_split:
                chosen_val_split = candidate
                break

    if chosen_val_split is not None and chosen_val_split in ds:
        val_grouped = group_hf_split_by_image(ds[chosen_val_split], split_name=chosen_val_split)
        split_info = {
            "dataset_name": dataset_name,
            "train_split_source": train_hf_split,
            "val_split_source": chosen_val_split,
            "used_fallback_split": False,
        }
        return train_grouped, val_grouped, split_info

    # Fallback if dataset provides only one split.
    all_groups = list(train_grouped.groups)
    rng = random.Random(seed)
    rng.shuffle(all_groups)
    train_size = int(train_split * len(all_groups))
    fallback_train_groups = all_groups[:train_size]
    fallback_val_groups = all_groups[train_size:]

    train_grouped = HFGroupedSplit(
        split_name=f"{train_hf_split}_train_fallback",
        hf_split=train_grouped.hf_split,
        schema=train_grouped.schema,
        groups=fallback_train_groups,
    )
    val_grouped = HFGroupedSplit(
        split_name=f"{train_hf_split}_val_fallback",
        hf_split=train_grouped.hf_split,
        schema=train_grouped.schema,
        groups=fallback_val_groups,
    )
    split_info = {
        "dataset_name": dataset_name,
        "train_split_source": train_hf_split,
        "val_split_source": "fallback_random_split",
        "used_fallback_split": True,
        "fallback_train_ratio": train_split,
    }
    return train_grouped, val_grouped, split_info


def _decode_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
        if isinstance(image_value.get("path"), str):
            return Image.open(image_value["path"]).convert("RGB")
    raise ValueError(f"Unsupported image value type: {type(image_value)}")


def _build_debug_path(split_name, image_key):
    return f"hf://{split_name}/{image_key}"


class HFFlickr8kUniqueImageDataset(Dataset):
    """Training dataset over unique images from a grouped HF split."""

    def __init__(self, grouped_split, image_transform=None, is_train=True):
        if image_transform:
            self.image_transform = image_transform
        elif is_train:
            self.image_transform = self._get_train_transform()
        else:
            self.image_transform = self._get_eval_transform()

        self.grouped_split = grouped_split
        self.groups = grouped_split.groups
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.6),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _get_eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        row = self.grouped_split.hf_split[group.row_index]
        image = _decode_image(row[self.grouped_split.schema.image_column])

        rng = random.Random(self.epoch * 1000000 + idx)
        cap_idx = rng.randint(0, len(group.captions) - 1)
        caption = group.captions[cap_idx]

        return {
            "image": self.image_transform(image),
            "caption": caption,
            "image_id": idx,
            "caption_id": cap_idx,
            "image_path": _build_debug_path(self.grouped_split.split_name, group.image_key),
        }


class HFFlickr8kCanonicalCaptionDataset(Dataset):
    """Evaluation dataset with one canonical caption per unique image."""

    def __init__(self, grouped_split, image_transform=None):
        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = self._get_eval_transform()

        self.grouped_split = grouped_split
        self.groups = grouped_split.groups

    def _get_eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        row = self.grouped_split.hf_split[group.row_index]
        image = _decode_image(row[self.grouped_split.schema.image_column])
        caption = group.captions[0]

        return {
            "image": self.image_transform(image),
            "caption": caption,
            "image_id": idx,
            "caption_id": 0,
            "image_path": _build_debug_path(self.grouped_split.split_name, group.image_key),
        }


class HFFlickr8kAllCaptionsDataset(Dataset):
    """Evaluation dataset with all captions per unique image."""

    def __init__(self, grouped_split, image_transform=None):
        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = self._get_eval_transform()

        self.grouped_split = grouped_split
        self.groups = grouped_split.groups
        self.flat_map = []
        for img_idx, group in enumerate(self.groups):
            for cap_idx in range(len(group.captions)):
                self.flat_map.append((img_idx, cap_idx))

    def _get_eval_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self):
        return len(self.flat_map)

    def __getitem__(self, i):
        img_idx, cap_idx = self.flat_map[i]
        group = self.groups[img_idx]
        row = self.grouped_split.hf_split[group.row_index]
        image = _decode_image(row[self.grouped_split.schema.image_column])
        caption = group.captions[cap_idx]

        return {
            "image": self.image_transform(image),
            "caption": caption,
            "image_id": img_idx,
            "caption_id": i,
            "image_path": _build_debug_path(self.grouped_split.split_name, group.image_key),
        }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
