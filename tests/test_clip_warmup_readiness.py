from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src import clip_train
from src.clip_gradient_stages import observational_state, ROOT
from src.clip_warmup_readiness import probe_objectives, projection_probe, gradient_vectors, vector_metrics


@pytest.fixture(autouse=True)
def isolate_rng():
    with observational_state(torch.nn.Identity()):
        yield


def test_head_gradient_matches_full_encoder_graph_without_mutation():
    torch.manual_seed(82)
    image_encoder, text_encoder = torch.nn.Linear(7, 6), torch.nn.Linear(7, 5)
    image_head, text_head = torch.nn.Linear(6, 4, bias=False), torch.nn.Linear(5, 4, bias=False)
    pooled_i, pooled_t = image_encoder(torch.randn(10, 7)), text_encoder(torch.randn(10, 7))
    scale = torch.tensor(4., requires_grad=True)
    params = tuple(image_head.parameters()) + tuple(text_head.parameters())
    for p in params:
        p.grad = torch.ones_like(p)
    old_params, old_grads = [p.detach().clone() for p in params], [p.grad.clone() for p in params]
    objectives = probe_objectives(F.normalize(image_head(pooled_i), dim=-1),
                                 F.normalize(text_head(pooled_t), dim=-1), scale)
    full = gradient_vectors(objectives, params)
    expected = vector_metrics({name: torch.cat([g.flatten() for g in grads]) for name, grads in full.items()})
    rows = projection_probe(image_head, text_head, pooled_i, pooled_t, scale)
    assert len(rows) == 4
    for key, value in expected.items():
        assert rows[0][key] == pytest.approx(value, rel=1e-6, abs=1e-7)
    for p, value, grad in zip(params, old_params, old_grads):
        assert torch.equal(p, value) and torch.equal(p.grad, grad)
    assert scale.grad is None
    assert all(p.grad is None for p in image_encoder.parameters())


def test_projection_native_gradient_predicts_finite_difference():
    torch.manual_seed(82)
    heads = torch.nn.Linear(6, 4, bias=False), torch.nn.Linear(5, 4, bias=False)
    i, t, scale = torch.randn(10, 6), torch.randn(10, 5), torch.tensor(3.)
    def objective():
        return probe_objectives(F.normalize(heads[0](i), dim=-1), F.normalize(heads[1](t), dim=-1), scale)
    initial = objective()
    params = tuple(heads[0].parameters()) + tuple(heads[1].parameters())
    aux_grads = torch.autograd.grad(initial['u8'], params, retain_graph=True)
    native_grads = torch.autograd.grad(initial['native_symmetric'], params)
    norm = torch.cat([g.flatten() for g in aux_grads]).norm()
    predicted = -sum((a * n).sum() for a, n in zip(aux_grads, native_grads)) / norm
    epsilon = 1e-4
    with torch.no_grad():
        for p, g in zip(params, aux_grads):
            p.add_(-epsilon * g / norm)
    actual = (objective()['native_symmetric'] - initial['native_symmetric']) / epsilon
    assert float(actual.detach()) == pytest.approx(float(predicted), abs=.006)


def test_undefined_cosines_are_null():
    vectors = {k: torch.zeros(4) for k in ('u8','real','native_symmetric','native_row','margin')}
    result = vector_metrics(vectors)
    assert result['cos_u8_native_symmetric'] is None
    assert result['margin_change_u8'] is None


def test_observer_captures_and_reconstructs_head_inputs(tmp_path, monkeypatch):
    import json
    from src import clip_gradient_stages
    from src.clip_warmup_readiness import WarmupObserver
    from src.clip_negative_gradient_geometry_randomized import build_partition_conditions, EXPECTED_PARTITIONS
    torch.manual_seed(7)
    model = torch.nn.Module()
    model.clip_model = torch.nn.Module()
    model.clip_model.visual_projection = torch.nn.Linear(6, 4, bias=False)
    model.clip_model.text_projection = torch.nn.Linear(5, 4, bias=False)
    model.get_logit_scale = lambda: torch.tensor(5.)
    image_pooled, text_pooled = torch.randn(20, 6), torch.randn(20, 5)
    metadata = [{'source_index': i, 'image_key': str(i), 'species_id': i % 4} for i in range(20)]
    @torch.inference_mode()
    def encode(*args):
        return (F.normalize(model.clip_model.visual_projection(image_pooled), dim=-1),
                F.normalize(model.clip_model.text_projection(text_pooled), dim=-1), metadata)
    monkeypatch.setattr(clip_gradient_stages, 'encode_dataset', encode)
    observer = WarmupObserver(tmp_path, {'save_features': True, 'top_k': 8,
        'holdout_size': 20, 'batch_size': 10, 'tangent_tolerance': 1e-5}, {'projection_batches_per_partition': 2})
    observer.stages = {0: 'pretrained'}
    observer.processor, observer.loader, observer.device = None, None, torch.device('cpu')
    observer.config = {'ot': {'enabled': False, 'alpha_max': .5, 'warmup_steps': 1000, 'ramp_steps': 1000}}
    observer.holdout_hash = 'fixture'
    observer.conditions, observer.partitions = build_partition_conditions(EXPECTED_PARTITIONS, 20, 10)
    observer(model=model, epoch=0, global_step=0)
    observer.finish()
    assert len(json.loads((tmp_path / 'projection_summary.json').read_text())) == 32
    saved = torch.load(tmp_path / '000000_pretrained/projection_inputs_and_heads.pt', weights_only=True)
    assert torch.equal(saved['pooled_images'], image_pooled)
    assert torch.equal(saved['pooled_texts'], text_pooled)
    assert not model.clip_model.visual_projection._forward_pre_hooks
    assert all(p.grad is None for p in model.parameters())


def test_short_execution_preserves_original_scheduler_horizon(monkeypatch):
    from transformers import AutoProcessor
    config = clip_train.load_training_config(ROOT / 'configs/hf_cub200_clip_vit_b32_baseline.yaml')
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *args: None)
    monkeypatch.setattr(clip_train, 'resolve_device', lambda config: (torch.device('cpu'), {}))
    monkeypatch.setattr(clip_train, 'build_clip_training_data', lambda *args, **kwargs: SimpleNamespace(train_loader=range(77)))
    monkeypatch.setattr(clip_train.CLIPEncoderBackend, 'from_pretrained', lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(clip_train, 'configure_clip_trainable_parameters', lambda *args: {})
    monkeypatch.setattr(clip_train, 'build_clip_optimizer', lambda *args: None)
    class StopAfterScheduler(Exception):
        pass
    captured = {}
    def scheduler(optimizer, **kwargs):
        captured.update(kwargs)
        raise StopAfterScheduler
    monkeypatch.setattr(clip_train, 'build_scheduler', scheduler)
    with pytest.raises(StopAfterScheduler):
        clip_train.run(config, stop_after_epochs=13)
    assert captured == {'warmup_steps': 100, 'total_steps': 3850}
    with pytest.raises(ValueError, match='stop_after_epochs'):
        clip_train.run(config, stop_after_epochs=51)
