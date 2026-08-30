"""Shared retrieval and species evaluation for both CLIP experiment arms."""

from contextlib import nullcontext

import torch

from src.clip_geometry_v2_metrics import compute_zero_shot_species_evaluation


def _autocast(device, mixed_precision):
    if device.type != "cuda" or mixed_precision == "none":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask"}
    }


@torch.inference_mode()
def encode_canonical_pairs(model, loader, device, mixed_precision):
    image_features = []
    text_features = []
    image_ids = []
    species_ids = []
    for batch in loader:
        with _autocast(device, mixed_precision):
            image_features.append(
                model.encode_images(
                    batch["pixel_values"].to(device, non_blocking=True)
                ).float().cpu()
            )
            text_features.append(
                model.encode_texts(_to_device(batch, device)).float().cpu()
            )
        image_ids.extend(batch["image_ids"].tolist())
        species_ids.extend(batch["species_ids"].tolist())
    return {
        "image_features": torch.cat(image_features),
        "text_features": torch.cat(text_features),
        "image_ids": image_ids,
        "species_ids": species_ids,
    }


@torch.inference_mode()
def encode_all_caption_texts(model, loader, device, mixed_precision):
    features = []
    image_ids = []
    for batch in loader:
        with _autocast(device, mixed_precision):
            features.append(
                model.encode_texts(_to_device(batch, device)).float().cpu()
            )
        image_ids.extend(batch["image_ids"].tolist())
    return torch.cat(features), image_ids


def chunked_bidirectional_retrieval(
    text_features,
    image_features,
    text_image_ids,
    *,
    chunk_size=512,
    device=None,
):
    """Exact recall without materializing the full all-caption score matrix."""
    text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    text_image_ids = torch.as_tensor(text_image_ids, dtype=torch.long)
    compute_device = torch.device(device or "cpu")
    image_features_device = image_features.to(compute_device)
    text_ranks = []
    for start in range(0, len(text_features), chunk_size):
        text = text_features[start : start + chunk_size].to(compute_device)
        targets = text_image_ids[start : start + chunk_size].to(compute_device)
        scores = text @ image_features_device.T
        true_scores = scores.gather(1, targets.unsqueeze(1))
        text_ranks.append(((scores > true_scores).sum(1) + 1).cpu())
    text_ranks = torch.cat(text_ranks)

    text_features_device = text_features.to(compute_device)
    text_image_ids_device = text_image_ids.to(compute_device)
    image_ranks = []
    for start in range(0, len(image_features), chunk_size):
        image = image_features[start : start + chunk_size].to(compute_device)
        scores = image @ text_features_device.T
        image_ids = torch.arange(
            start, start + len(image), device=compute_device
        ).unsqueeze(1)
        positives = text_image_ids_device.unsqueeze(0) == image_ids
        if not positives.any(dim=1).all():
            raise ValueError("At least one image has no positive caption")
        best_positive = scores.masked_fill(~positives, float("-inf")).max(1).values
        image_ranks.append(
            ((scores > best_positive.unsqueeze(1)).sum(1) + 1).cpu()
        )
    image_ranks = torch.cat(image_ranks)

    def recalls(ranks):
        return {
            f"r_at_{k}": float((ranks <= k).float().mean().item() * 100)
            for k in (1, 5, 10)
        }

    return {
        "text_to_image": recalls(text_ranks),
        "image_to_text": recalls(image_ranks),
    }


@torch.inference_mode()
def encode_species_prompts(
    model, processor, species_names, prompt_template, device, mixed_precision
):
    species_ids = sorted(species_names)
    prompts = [
        prompt_template.format(species_name=species_names[index])
        for index in species_ids
    ]
    processed = processor(
        text=prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_batch = {
        key: value.to(device)
        for key, value in processed.items()
        if key in {"input_ids", "attention_mask"}
    }
    with _autocast(device, mixed_precision):
        features = model.encode_texts(text_batch).float().cpu()
    return features, species_ids


@torch.inference_mode()
def evaluate_clip(
    model,
    processor,
    data,
    device,
    *,
    mixed_precision,
    prompt_template,
    retrieval_chunk_size,
):
    """Evaluate canonical/all-caption retrieval and fixed-prompt species metrics."""
    was_training = model.training
    model.eval()
    canonical = encode_canonical_pairs(
        model, data.validation_canonical_loader, device, mixed_precision
    )
    all_text, all_image_ids = encode_all_caption_texts(
        model, data.validation_all_captions_loader, device, mixed_precision
    )
    species_text, prompt_species_ids = encode_species_prompts(
        model,
        processor,
        data.species_names,
        prompt_template,
        device,
        mixed_precision,
    )
    canonical_retrieval = chunked_bidirectional_retrieval(
        canonical["text_features"],
        canonical["image_features"],
        canonical["image_ids"],
        chunk_size=retrieval_chunk_size,
        device=device,
    )
    all_caption_retrieval = chunked_bidirectional_retrieval(
        all_text,
        canonical["image_features"],
        all_image_ids,
        chunk_size=retrieval_chunk_size,
        device=device,
    )
    species = compute_zero_shot_species_evaluation(
        canonical["image_features"],
        species_text,
        canonical["species_ids"],
        prompt_species_ids,
    )
    if was_training:
        model.train()
    return {
        "canonical_retrieval": canonical_retrieval,
        "all_caption_retrieval": all_caption_retrieval,
        "species": species,
        "validation_image_count": len(canonical["image_features"]),
        "validation_caption_count": len(all_text),
        "species_prompt": prompt_template,
    }
