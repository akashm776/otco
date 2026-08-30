"""Encoder adapter for jointly pretrained Hugging Face CLIP checkpoints.

The existing :class:`model.model.OTLIP` implementation intentionally remains
untouched.  This adapter gives diagnostics and future training code the same
high-level operations (encode images, encode text, compare embeddings) without
making the ResNet/DistilBERT path depend on CLIP-specific preprocessing or
temperature semantics.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CLIPSimilarityOutput:
    """Normalized embeddings and their two similarity representations."""

    raw_similarity: torch.Tensor
    logits: torch.Tensor
    image_features: torch.Tensor
    text_features: torch.Tensor
    logit_scale: torch.Tensor


def _feature_tensor(output):
    """Accept the tensor and model-output forms used across transformers versions."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Could not extract an embedding tensor from {type(output)!r}")


class CLIPEncoderBackend(nn.Module):
    """Thin, model-independent contract around a pretrained CLIP model.

    CLIP's learned multimodal projections are retained.  Both encoding methods
    return unit vectors, raw similarities are cosine similarities, and
    contrastive logits are produced exactly once using CLIP's learned
    ``exp(logit_scale)``.  Keeping these quantities separate prevents accidental
    double temperature scaling in later OTCO experiments.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        from transformers import CLIPModel

        return cls(CLIPModel.from_pretrained(checkpoint, **kwargs))

    def freeze(self):
        """Freeze the complete checkpoint and switch to deterministic eval mode."""
        self.clip_model.requires_grad_(False)
        self.eval()
        return self

    def encode_images(self, pixel_values):
        features = _feature_tensor(
            self.clip_model.get_image_features(pixel_values=pixel_values)
        )
        return F.normalize(features, dim=-1)

    def encode_texts(self, text_batch):
        features = _feature_tensor(self.clip_model.get_text_features(**text_batch))
        return F.normalize(features, dim=-1)

    def get_logit_scale(self):
        """Return CLIP's positive learned scale, not a second temperature."""
        logit_scale = getattr(self.clip_model, "logit_scale", None)
        if logit_scale is None:
            raise AttributeError("The CLIP checkpoint does not expose logit_scale")
        return logit_scale.exp()

    def compare(self, image_features, text_features):
        raw_similarity = text_features @ image_features.T
        logit_scale = self.get_logit_scale()
        return CLIPSimilarityOutput(
            raw_similarity=raw_similarity,
            logits=raw_similarity * logit_scale,
            image_features=image_features,
            text_features=text_features,
            logit_scale=logit_scale,
        )

    def forward(self, pixel_values, text_batch):
        image_features = self.encode_images(pixel_values)
        text_features = self.encode_texts(text_batch)
        return self.compare(image_features, text_features)
