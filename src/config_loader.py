from copy import deepcopy
from pathlib import Path

import yaml


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at '{path}' must contain a top-level mapping.")
    return data


def _resolve_path(base_path, maybe_relative_path):
    path = Path(maybe_relative_path)
    if path.is_absolute():
        return path
    return (base_path.parent / path).resolve()


def load_run_config(config_path):
    config_path = Path(config_path).resolve()
    run_cfg = _read_yaml(config_path)

    experiment_section = run_cfg.get("experiment", {})
    if not isinstance(experiment_section, dict):
        raise ValueError("'experiment' section must be a mapping.")

    exp_name = experiment_section.get("name")
    if not exp_name:
        raise ValueError("Missing required config key: experiment.name")

    experiments_file = experiment_section.get("experiments_file", "experiments.yaml")
    experiments_path = _resolve_path(config_path, experiments_file)
    experiments_yaml = _read_yaml(experiments_path)
    experiments = experiments_yaml.get("experiments", {})
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"No experiments found under 'experiments' in {experiments_path}")
    if exp_name not in experiments:
        available = ", ".join(sorted(experiments.keys()))
        raise ValueError(f"Unknown experiment '{exp_name}' in {experiments_path}. Available: {available}")

    experiment_config = deepcopy(experiments[exp_name])
    overrides = experiment_section.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("experiment.overrides must be a mapping.")
    experiment_config.update(overrides)
    experiment_config["experiment_name"] = exp_name

    model_section = run_cfg.get("model", {})
    if model_section is None:
        model_section = {}
    if not isinstance(model_section, dict):
        raise ValueError("'model' section must be a mapping.")
    if model_section.get("vision"):
        experiment_config["model_vision"] = model_section["vision"]
    if model_section.get("text"):
        experiment_config["model_text"] = model_section["text"]

    dataset_section = run_cfg.get("dataset", {})
    if dataset_section is None:
        dataset_section = {}
    if not isinstance(dataset_section, dict):
        raise ValueError("'dataset' section must be a mapping.")
    dataset_backend = dataset_section.get("backend", "local_flickr8k")
    if dataset_backend not in {"local_flickr8k", "hf_flickr8k", "hf_flickr30k"}:
        raise ValueError("dataset.backend must be one of: local_flickr8k, hf_flickr8k")

    run_section = run_cfg.get("run", {})
    if run_section is None:
        run_section = {}
    if not isinstance(run_section, dict):
        raise ValueError("'run' section must be a mapping.")

    return {
        "config_path": str(config_path),
        "experiments_path": str(experiments_path),
        "experiment_name": exp_name,
        "experiment_config": experiment_config,
        "dataset_backend": dataset_backend,
        "dataset": dataset_section,
        "run": run_section,
    }


_DIAGNOSTIC_DEFAULTS = {
    "checkpoint_path": "checkpoints/baseline/best_model.pt",
    "vision_model_name": "microsoft/resnet-50",
    "text_model_name": "distilbert-base-uncased",
    "train_split": 0.8,
    "eval_split": "val",
    "retrieval_pool_split": "val",
    "top_k": 5,
    "batch_size": 64,
    "num_workers": 4,
    "epsilon": 0.07,
    "max_length": 77,
    "max_captions": 100,
    "caption_stride": 1,
    "canonical_only": False,
    "output_dir": "outputs/ot_diagnostic",
    "cache_image_embs": True,
}


def load_diagnostic_config(config_path):
    config_path = Path(config_path).resolve()
    cfg_yaml = _read_yaml(config_path)

    diagnostic_section = cfg_yaml.get("diagnostic", {})
    if diagnostic_section is None:
        diagnostic_section = {}
    if not isinstance(diagnostic_section, dict):
        raise ValueError("'diagnostic' section must be a mapping.")

    model_section = cfg_yaml.get("model", {})
    if model_section is None:
        model_section = {}
    if not isinstance(model_section, dict):
        raise ValueError("'model' section must be a mapping.")

    cfg = deepcopy(_DIAGNOSTIC_DEFAULTS)
    cfg.update(diagnostic_section)

    if model_section.get("vision"):
        cfg["vision_model_name"] = model_section["vision"]
    if model_section.get("text"):
        cfg["text_model_name"] = model_section["text"]

    if cfg["eval_split"] not in {"train", "val"}:
        raise ValueError("diagnostic.eval_split must be one of: train, val")
    if cfg["retrieval_pool_split"] not in {"train", "val"}:
        raise ValueError("diagnostic.retrieval_pool_split must be one of: train, val")
    if cfg["top_k"] <= 0:
        raise ValueError("diagnostic.top_k must be > 0")
    if cfg["batch_size"] <= 0:
        raise ValueError("diagnostic.batch_size must be > 0")
    if cfg["num_workers"] < 0:
        raise ValueError("diagnostic.num_workers must be >= 0")
    if cfg["caption_stride"] <= 0:
        raise ValueError("diagnostic.caption_stride must be > 0")

    return {
        "config_path": str(config_path),
        "diagnostic_config": cfg,
    }
