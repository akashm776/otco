from data.hf_flickr8k_dataset import (
    HFFlickr8kAllCaptionsDataset as HFCUB200AllCaptionsDataset,
    HFFlickr8kCanonicalCaptionDataset as HFCUB200CanonicalCaptionDataset,
    HFFlickr8kUniqueImageDataset as HFCUB200UniqueImageDataset,
    HFColumnSchema,
    HFGroupedImage,
    HFGroupedSplit,
    seed_worker,
)

try:
    from datasets import load_dataset as _load_dataset
except Exception:
    _load_dataset = None

__all__ = [
    "HFCUB200UniqueImageDataset",
    "HFCUB200AllCaptionsDataset",
    "HFCUB200CanonicalCaptionDataset",
    "load_hf_cub200_splits",
    "seed_worker",
]

_DATASET_NAME = "alkzar90/CC6204-Hackaton-Cub-Dataset"
# description column holds 10 captions joined by \n; file_name is the stable image key
_SCHEMA = HFColumnSchema(image_column="image", caption_column="description", image_id_column="file_name")


def _build_grouped_split(hf_split, split_name):
    groups = []
    for row_index in range(len(hf_split)):
        example = hf_split[row_index]
        image_key = str(example["file_name"])
        captions = [c.strip() for c in example["description"].split("\n") if c.strip()]
        if not captions:
            continue
        groups.append(HFGroupedImage(row_index=row_index, image_key=image_key, captions=captions))
    return HFGroupedSplit(split_name=split_name, hf_split=hf_split, schema=_SCHEMA, groups=groups)


def load_hf_cub200_splits(
    dataset_name=_DATASET_NAME,
    train_hf_split="train",
    val_hf_split="test",
):
    if _load_dataset is None:
        raise ImportError("datasets library is required. Install with: pip install datasets")

    ds = _load_dataset(dataset_name)
    available = list(ds.keys())

    if train_hf_split not in ds:
        raise ValueError(f"Train split '{train_hf_split}' not found. Available: {available}")
    if val_hf_split not in ds:
        raise ValueError(f"Val split '{val_hf_split}' not found. Available: {available}")

    train_grouped = _build_grouped_split(ds[train_hf_split], split_name=train_hf_split)
    val_grouped = _build_grouped_split(ds[val_hf_split], split_name=val_hf_split)

    split_info = {
        "dataset_name": dataset_name,
        "train_split_source": train_hf_split,
        "val_split_source": val_hf_split,
        "used_fallback_split": False,
    }
    return train_grouped, val_grouped, split_info
