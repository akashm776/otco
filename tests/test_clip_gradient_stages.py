import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from model.clip_training import scheduled_ot_alpha
from src import clip_gradient_stages as stages
from src.clip_negative_gradient_geometry_randomized import build_partition_conditions, EXPECTED_PARTITIONS
from src.clip_train import load_training_config, train_epoch


@pytest.fixture(autouse=True)
def isolate_test_rng():
    with stages.observational_state(torch.nn.Identity()):
        yield


def test_exact_schedule_boundaries_and_existing_arm_configs():
    assert stages.stage_steps(1000, 1000, 3850) == {
        0: "pretrained", 1001: "before_auxiliary", 2001: "after_ramp", 3850: "late"}
    alpha = lambda step: scheduled_ot_alpha(step, alpha_max=.5, warmup_steps=1000, ramp_steps=1000)
    assert alpha(1000) == 0 < alpha(1001)
    assert alpha(1999) < alpha(2000) == .5
    protocol = yaml.safe_load((stages.ROOT / "configs/clip_gradient_stages.yaml").read_text())
    for path in protocol["arms"].values():
        config = load_training_config(stages.ROOT / path)
        assert config["training"]["epochs"] == 50
        assert config["ot"]["warmup_steps"] == 1000
        assert config["ot"]["ramp_steps"] == 1000
    with pytest.raises(ValueError):
        stages.stage_steps(1000, 1000, 1500)


def test_observational_state_restores_rng_modes_and_grads_on_failure():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout())
    model.train()
    model[0].eval()
    model(torch.ones(2, 4)).sum().backward()
    params = [(p.detach().clone(), p.grad.clone()) for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    expected = random.random(), np.random.random(), torch.rand(1)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    with pytest.raises(RuntimeError), stages.observational_state(model):
        assert not any(m.training for m in model.modules())
        random.random(), np.random.random(), torch.rand(100)
        raise RuntimeError("diagnostic failure")
    actual = random.random(), np.random.random(), torch.rand(1)
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert [m.training for m in model.modules()] == modes
    for parameter, (value, grad) in zip(model.parameters(), params):
        assert torch.equal(parameter, value) and torch.equal(parameter.grad, grad)


def test_observer_runs_real_gradient_math_and_writes_fixed_paired_stages(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(72)
    features = [F.normalize(torch.randn(20, 6, generator=generator), dim=-1) for _ in range(2)]
    metadata = [{"source_index": i + 30, "image_key": f"bird/{i}", "species_id": i % 4}
                for i in range(20)]
    @torch.inference_mode()
    def encode(*args):
        return features[0].clone(), features[1].clone(), copy.deepcopy(metadata)
    monkeypatch.setattr(stages, "encode_dataset", encode)
    model = torch.nn.Linear(6, 6)
    model.get_logit_scale = lambda: torch.tensor(10.)
    model(torch.ones(1, 6)).sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    observer = stages.GradientStageObserver(tmp_path, {
        "save_features": True, "top_k": 8, "tangent_tolerance": 1e-5}, "toy")
    observer.stages = {0: "pretrained", 1001: "before_auxiliary"}
    observer.processor, observer.loader, observer.device = None, None, torch.device("cpu")
    observer.config = {"ot": {"enabled": True, "alpha_max": .5,
                               "warmup_steps": 1000, "ramp_steps": 1000}}
    observer.holdout_hash = "test-holdout"
    observer.conditions, observer.partitions = build_partition_conditions(EXPECTED_PARTITIONS, 20, 10)
    observer(model=model, epoch=0, global_step=0)
    observer(model=model, epoch=1, global_step=1)  # no extra measurement
    with pytest.raises(AssertionError):
        observer.finish()
    observer(model=model, epoch=14, global_step=1001)
    observer.finish()
    with pytest.raises(ValueError, match="Repeated"):
        observer(model=model, epoch=14, global_step=1001)
    reports = json.loads((tmp_path / "stage_summary.json").read_text())
    assert len(reports) == 2
    assert reports[0]["metadata_sha256"] == reports[1]["metadata_sha256"]
    assert reports[0]["features_and_scale_sha256"] == reports[1]["features_and_scale_sha256"]
    assert reports[1]["previous_update_scheduled_alpha"] == 0
    assert reports[1]["next_update_scheduled_alpha"] > 0
    assert reports[1]["transition_from_previous_stage"]["mean_native_alignment_u8_change"] == 0
    assert reports[0]["compact_by_partition"] == reports[1]["compact_by_partition"]
    saved = torch.load(tmp_path / "000000_pretrained/features.pt", weights_only=True)
    assert torch.equal(saved["image_features"], features[0])
    assert model.training
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    assert len((tmp_path / "001001_before_auxiliary/per_query.csv").read_text().splitlines()) == 81


def test_paired_transition_rejects_changed_identity():
    row = {"partition_name": "shuffle_seed_42", "query_index": 0,
           "source_index": 9, "image_key": "bird", "native_alignment_u8": -.1,
           "margin_directional_change_u8": -.2}
    new = {**row, "native_alignment_u8": .2, "margin_directional_change_u8": .3}
    result = stages.paired_transitions([row], [new])
    assert result["fraction_u8_margin_nonpositive_to_positive"] == 1
    with pytest.raises(ValueError):
        stages.paired_transitions([row], [{**new, "source_index": 10}])


def test_undefined_directions_are_retained_not_fabricated():
    from src.clip_negative_gradient_geometry import compute_batch_gradient_geometry
    features = F.normalize(torch.ones(10, 6), dim=-1)
    result = compute_batch_gradient_geometry(features, features, 10., top_k=8,
                                            record_undefined=True)
    assert len(result["rows"]) == 10
    assert all(not r["gradient_metrics_valid"] for r in result["rows"])
    compact = stages.compact_metrics(result["rows"])
    assert compact["valid_query_count"] == 0
    assert compact["undefined_query_count"] == 10
    assert compact["joint_cosine"] is None


def test_plot_outputs_and_mismatched_holdout_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    compact = {key: .2 for key in stages.METRICS}
    compact.update(fraction_u8_jointly_useful=.4, valid_query_count=1024, undefined_query_count=0)
    reports = [{"arm": "baseline", "stage": name, "completed_updates": step,
                "holdout_sha256": "holdout", "metadata_sha256": "metadata",
                "partition_sha256": "partition",
                "compact_by_partition": {spec["name"]: compact for spec in EXPECTED_PARTITIONS}}
               for step, name in stages.stage_steps(1000, 1000, 3850).items()]
    target = tmp_path / "baseline/diagnostics"
    target.mkdir(parents=True)
    (target / "stage_summary.json").write_text(json.dumps(reports))
    stages.plot_results(tmp_path)
    for name in ("gradient_stage_trends.png", "gradient_stage_trends.svg", "gradient_stage_comparison.csv"):
        assert (tmp_path / name).stat().st_size > 100
    target2 = tmp_path / "uniform_top8/diagnostics"
    target2.mkdir(parents=True)
    reports[0]["holdout_sha256"] = "different"
    (target2 / "stage_summary.json").write_text(json.dumps(reports))
    with pytest.raises(ValueError, match="Cannot compare"):
        stages.plot_results(tmp_path)


def test_training_with_observations_is_identical_to_without():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clip_model = torch.nn.Module()
            self.clip_model.logit_scale = torch.nn.Parameter(torch.tensor(1.))
            self.projection = torch.nn.Linear(4, 4)
            self.dropout = torch.nn.Dropout(.3)
        def get_logit_scale(self):
            return self.clip_model.logit_scale.exp()
        def forward(self, pixels, text):
            return self.dropout(self.projection(pixels)) * self.get_logit_scale()
    def execute(observe):
        torch.manual_seed(12)
        model = Model()
        batches = [{"pixel_values": torch.randn(3, 4), "input_ids": torch.ones(3, 2),
                    "attention_mask": torch.ones(3, 2), "species_ids": torch.arange(3)}
                   for _ in range(4)]
        data = SimpleNamespace(train_dataset=SimpleNamespace(set_epoch=lambda epoch: None), train_loader=batches)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
        seen = []
        def diagnostic(**kwargs):
            seen.append(kwargs["global_step"])
            with stages.observational_state(kwargs["model"]):
                torch.rand(100)
                with torch.no_grad():
                    kwargs["model"](torch.ones(3, 4), None)
        objective = lambda output, **kwargs: SimpleNamespace(
            total_loss=output.square().mean(), weighted_ot_loss=None, metrics={})
        result = train_epoch(model, objective, data, optimizer, scheduler, torch.device("cpu"),
                             {"training": {"mixed_precision": "none"},
                              "optimizer": {"max_gradient_norm": 1.}},
                             epoch=1, global_step=0, remaining_gradient_diagnostics=0,
                             step_observer=diagnostic if observe else None)
        return copy.deepcopy(model.state_dict()), torch.get_rng_state(), result, seen
    plain, watched = execute(False), execute(True)
    assert all(torch.equal(plain[0][k], watched[0][k]) for k in plain[0])
    assert torch.equal(plain[1], watched[1])
    assert plain[2] == watched[2]
    assert watched[3] == [1, 2, 3, 4]
