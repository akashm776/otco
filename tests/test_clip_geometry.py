"""Unit tests for the frozen CLIP adapter and geometry diagnostic."""

import math

import torch
import torch.nn as nn

from model.clip_backend import CLIPEncoderBackend
from model.loss import SoftmaxMixLoss
from src.clip_geometry_diagnostic import stratified_holdout_indices
from src.clip_geometry_metrics import (
    build_sparse_transport_plan,
    compute_geometry_diagnostics,
)


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.weight = nn.Parameter(torch.ones(()))

    def get_image_features(self, pixel_values):
        return pixel_values * self.weight

    def get_text_features(self, input_ids, attention_mask=None):
        return input_ids.float() * self.weight


def test_clip_backend_normalizes_and_scales_once():
    backend = CLIPEncoderBackend(FakeCLIP()).freeze()
    images = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    texts = {"input_ids": torch.tensor([[6, 8], [2, 0]])}
    output = backend(images, texts)

    assert torch.allclose(output.image_features.norm(dim=1), torch.ones(2))
    assert torch.allclose(output.text_features.norm(dim=1), torch.ones(2))
    assert torch.allclose(output.logits, output.raw_similarity * 10.0)
    assert all(not parameter.requires_grad for parameter in backend.parameters())


def test_transport_excludes_positives_and_satisfies_dense_off_diagonal_marginals():
    logits = torch.tensor(
        [
            [5.0, 1.0, 2.0, 3.0],
            [1.0, 5.0, 3.0, 2.0],
            [2.0, 3.0, 5.0, 1.0],
            [3.0, 2.0, 1.0, 5.0],
        ]
    )
    positives = torch.eye(4, dtype=torch.bool)
    result = build_sparse_transport_plan(
        logits, positives, top_k=3, ot_eps=0.7, sinkhorn_iters=100
    )

    assert not result.support_mask[positives].any()
    assert torch.all(result.plan[positives] == 0)
    assert torch.allclose(result.plan.sum(1), torch.full((4,), 0.25), atol=1e-4)
    assert torch.allclose(result.plan.sum(0), torch.full((4,), 0.25), atol=1e-4)


def test_transport_matches_existing_ot_mix_implementation():
    torch.manual_seed(7)
    count = 8
    logits = torch.randn(count, count)
    dummy_features = torch.randn(count, 4)
    existing_loss = SoftmaxMixLoss(
        top_k=3, ot_eps=0.7, sinkhorn_iters=30
    )
    existing_plan, existing_support = existing_loss._make_plan(
        dummy_features, dummy_features, logits
    )
    diagnostic = build_sparse_transport_plan(
        logits,
        torch.eye(count, dtype=torch.bool),
        top_k=3,
        ot_eps=0.7,
        sinkhorn_iters=30,
    )

    assert torch.equal(diagnostic.support_mask, existing_support)
    assert torch.allclose(diagnostic.plan, existing_plan)


def test_geometry_reports_rank_gap_species_and_positive_exclusion():
    image_features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    text_features = image_features.clone()
    diagnostics = compute_geometry_diagnostics(
        image_features=image_features,
        text_features=text_features,
        logit_scale=10.0,
        image_ids=[10, 11, 12, 13],
        species_ids=["a", "a", "b", "b"],
        top_k=3,
        ot_eps=0.7,
        sinkhorn_iters=100,
    )

    selected = diagnostics["selected_indices"]
    assert torch.all(selected != torch.arange(4))
    assert torch.all(diagnostics["positive_selected_gap"] > 0)
    assert torch.all(diagnostics["selected_rank"] >= 1)
    assert diagnostics["same_species_hardest"].all()
    assert torch.isfinite(diagnostics["coupling_entropy"]).all()
    assert torch.all(
        (diagnostics["normalized_coupling_entropy"] >= 0)
        & (diagnostics["normalized_coupling_entropy"] <= 1.0001)
    )


def test_stratified_holdout_is_deterministic_and_balanced():
    labels = ["a"] * 10 + ["b"] * 10 + ["c"] * 10
    first = stratified_holdout_indices(
        labels, fraction=0.5, seed=42, max_samples=9
    )
    second = stratified_holdout_indices(
        labels, fraction=0.5, seed=42, max_samples=9
    )
    selected_labels = [labels[index] for index in first]

    assert first == second
    assert len(first) == 9
    assert {label: selected_labels.count(label) for label in set(labels)} == {
        "a": 3,
        "b": 3,
        "c": 3,
    }
