"""Focused tests for frozen CLIP negative-gradient geometry."""

from pathlib import Path
import inspect

import pytest
import torch
import torch.nn.functional as F

from model.clip_training import clip_relative_denominator_loss
from src.clip_negative_gradient_geometry import (
    compute_batch_gradient_geometry,
    load_config,
)
from src.clip_negative_gradient_geometry_metrics import (
    cosine_alignment,
    gradient_footprint,
    margin_directional_change,
)


ROOT = Path(__file__).resolve().parents[1]


def _features(batch_size=10, dimension=6):
    generator = torch.Generator().manual_seed(123)
    images = F.normalize(
        torch.randn(batch_size, dimension, generator=generator), dim=-1
    )
    texts = F.normalize(
        images + 0.1 * torch.randn(batch_size, dimension, generator=generator),
        dim=-1,
    )
    return images, texts


def _result():
    images, texts = _features()
    return compute_batch_gradient_geometry(
        images,
        texts,
        torch.tensor(14.0),
        species_ids=[index % 3 for index in range(images.shape[0])],
        top_k=8,
    )


def test_u8_support_and_hardest_real_exclude_diagonal_positives():
    result = _result()
    support = result["support_mask"]
    assert torch.all(support.sum(1) == 8)
    assert not torch.any(support.diagonal())
    for row in result["rows"]:
        assert row["hardest_real_index"] != row["query_index_within_batch"]


def test_u8_weights_are_exact_uniform_one_eighth_on_support():
    result = _result()
    support = result["support_mask"]
    weights = result["uniform_weights"]
    assert torch.all(weights[~support] == 0)
    assert torch.all(weights[support] == torch.tensor(1 / 8, dtype=weights.dtype))
    assert torch.all(weights.sum(1) == 1)


def test_u8_synthetic_is_exact_normalized_uniform_feature_average():
    images, texts = _features()
    result = compute_batch_gradient_geometry(images, texts, 14.0, top_k=8)
    expected = F.normalize(result["uniform_weights"] @ images, dim=-1)
    assert torch.allclose(result["u8_synthetic"], expected)


def test_known_hardest_real_indices_are_selected_deterministically():
    images = torch.eye(4)
    texts = F.normalize(
        torch.tensor(
            [
                [1.0, 0.9, 0.1, 0.0],
                [0.8, 1.0, 0.2, 0.1],
                [0.1, 0.7, 1.0, 0.6],
                [0.6, 0.1, 0.5, 1.0],
            ]
        ),
        dim=-1,
    )
    # Use top_k=2 because this fixture has only three eligible negatives.
    result = compute_batch_gradient_geometry(images, texts, 2.0, top_k=2)
    assert [row["hardest_real_index"] for row in result["rows"]] == [1, 0, 1, 0]


def test_embedding_gradients_are_live_and_local_model_parameter_stays_frozen():
    encoder = torch.nn.Linear(6, 6, bias=False).requires_grad_(False)
    inputs = torch.randn(10, 6, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        encoded = encoder(inputs)
    images = F.normalize(encoded, dim=-1)
    texts = F.normalize(images + 0.01, dim=-1)
    result = compute_batch_gradient_geometry(images, texts, 14.0, top_k=8)
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(row["query_gradient_norm_u8"] > 0 for row in result["rows"])
    assert all(row["query_gradient_norm_real"] > 0 for row in result["rows"])
    assert all(row["image_gradient_norm_u8"] > 0 for row in result["rows"])
    assert all(row["image_gradient_norm_real"] > 0 for row in result["rows"])


def test_diagnostic_query_and_image_gradients_are_tangent_to_unit_sphere():
    result = _result()
    audit = result["tangent_audit"]
    assert audit["max_abs_query_gradient_dot_embedding"] < 1e-5
    assert audit["max_abs_image_gradient_dot_embedding"] < 1e-5


def test_normalization_path_removes_radial_gradient_component():
    leaf = F.normalize(
        torch.tensor([2.0, -1.0, 0.5], dtype=torch.float64), dim=0
    ).requires_grad_(True)
    embedding = F.normalize(leaf, dim=0)
    ambient_gradient = torch.tensor(
        [1.5, 0.25, -0.75], dtype=torch.float64
    )
    loss = torch.dot(embedding, ambient_gradient)
    tangent_gradient = torch.autograd.grad(loss, leaf)[0]
    expected = ambient_gradient - torch.dot(
        ambient_gradient, embedding.detach()
    ) * embedding.detach()

    assert torch.allclose(tangent_gradient, expected, atol=1e-12, rtol=1e-12)
    assert torch.dot(tangent_gradient, embedding.detach()).abs() < 1e-12


def test_same_candidate_has_identical_query_gradient_direction():
    queries = F.normalize(torch.tensor([[1.0, 0.2]]), dim=-1).requires_grad_(True)
    images = F.normalize(
        torch.tensor([[0.8, 0.6], [0.8, 0.6], [-0.2, 1.0]]), dim=-1
    ).requires_grad_(True)
    base = queries @ images.T
    candidate_a = queries @ images[0]
    candidate_b = queries @ images[1]
    loss_a, _, _ = clip_relative_denominator_loss(base, candidate_a)
    loss_b, _, _ = clip_relative_denominator_loss(base, candidate_b)
    grad_a = torch.autograd.grad(loss_a, queries, retain_graph=True)[0]
    grad_b = torch.autograd.grad(loss_b, queries)[0]
    assert cosine_alignment(grad_a, grad_b).item() == pytest.approx(1.0, abs=1e-6)


def test_gradient_cosine_is_invariant_to_positive_rescaling():
    left = torch.tensor([1.0, -2.0, 3.0])
    right = torch.tensor([-1.0, 4.0, 2.0])
    assert cosine_alignment(left, right) == pytest.approx(
        cosine_alignment(7 * left, 0.2 * right).item()
    )


def test_effective_gradient_support_matches_known_patterns():
    one = torch.zeros(8, 2)
    one[3] = torch.tensor([2.0, 0.0])
    eight = torch.ones(8, 1)
    assert gradient_footprint(one)["effective_support"].item() == pytest.approx(1.0)
    assert gradient_footprint(eight)["effective_support"].item() == pytest.approx(8.0)


def test_margin_directional_derivative_matches_finite_difference():
    point = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
    margin = point[0].square() + 3 * point[1]
    gradient = torch.autograd.grad(margin, point)[0]
    auxiliary = torch.tensor([2.0, -1.0], dtype=torch.float64)
    analytic = margin_directional_change(gradient, auxiliary)
    direction = -auxiliary / auxiliary.norm()
    step = 1e-4
    moved = point.detach() + step * direction
    moved_margin = moved[0].square() + 3 * moved[1]
    finite = (moved_margin - margin.detach()) / step
    assert analytic.item() == pytest.approx(finite.item(), abs=2e-4)


def test_config_is_strictly_frozen_and_has_no_training_components():
    config = load_config(
        ROOT / "configs/hf_cub200_clip_negative_gradient_geometry.yaml"
    )
    assert config["diagnostic"] == {"batch_size": 64, "top_k": 8}
    assert "optimizer" not in config
    assert "scheduler" not in config
    assert "training" not in config


def test_diagnostic_source_has_no_optimizer_scheduler_or_backward_step():
    from src import clip_negative_gradient_geometry as diagnostic

    source = inspect.getsource(diagnostic)
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source
    assert ".backward(" not in source
