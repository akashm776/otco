from data.hf_flickr8k_dataset import (
    HFFlickr8kAllCaptionsDataset as HFFlickr30kAllCaptionsDataset,
    HFFlickr8kCanonicalCaptionDataset as HFFlickr30kCanonicalCaptionDataset,
    HFFlickr8kUniqueImageDataset as HFFlickr30kUniqueImageDataset,
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
    "HFFlickr30kUniqueImageDataset",
    "HFFlickr30kAllCaptionsDataset",
    "HFFlickr30kCanonicalCaptionDataset",
    "load_hf_flickr30k_splits",
    "seed_worker",
]

_DATASET_NAME = "clip-benchmark/wds_flickr30k"
_SCHEMA = HFColumnSchema(image_column="jpg", caption_column="txt", image_id_column="__key__")


def _build_grouped_split(hf_split, split_name):
    groups = []
    for row_index in range(len(hf_split)):
        example = hf_split[row_index]
        image_key = str(example["__key__"])
        captions = [c for c in example["txt"].split("\n") if c.strip()]
        if not captions:
            continue
        groups.append(HFGroupedImage(row_index=row_index, image_key=image_key, captions=captions))
    return HFGroupedSplit(split_name=split_name, hf_split=hf_split, schema=_SCHEMA, groups=groups)


def load_hf_flickr30k_splits(
    dataset_name=_DATASET_NAME,
    train_hf_split="train",
    val_hf_split="test",
    train_split=0.9,
    seed=42,
):
    if _load_dataset is None:
        raise ImportError("datasets library is required. Install with: pip install datasets")

    ds = _load_dataset(dataset_name)
    available = list(ds.keys())

    if train_hf_split not in ds:
        raise ValueError(f"Train split '{train_hf_split}' not found. Available: {available}")

    train_grouped = _build_grouped_split(ds[train_hf_split], split_name=train_hf_split)

    chosen_val = val_hf_split
    if chosen_val is None:
        for candidate in ("test", "validation", "val"):
            if candidate in ds and candidate != train_hf_split:
                chosen_val = candidate
                break

    if chosen_val is not None and chosen_val in ds:
        val_grouped = _build_grouped_split(ds[chosen_val], split_name=chosen_val)
        split_info = {
            "dataset_name": dataset_name,
            "train_split_source": train_hf_split,
            "val_split_source": chosen_val,
            "used_fallback_split": False,
        }
        return train_grouped, val_grouped, split_info

    raise ValueError(
        f"No validation split found. Available splits: {available}. "
        "Pass val_hf_split explicitly."
    )
