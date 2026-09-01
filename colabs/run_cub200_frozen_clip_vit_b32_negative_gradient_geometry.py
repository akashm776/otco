"""Run the frozen CLIP negative-gradient-geometry diagnostic on A100 Colab."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from google.colab import files, userdata


EXPERIMENT_NAME = "cub200_frozen_clip_vit_b32_negative_gradient_geometry"
CONFIG_FILE = "configs/hf_cub200_clip_negative_gradient_geometry.yaml"
PINNED_COMMIT = "35ca04e26c4d7dc4c7b9398995b65c0eb5bdcffb"
REPO_DIR = Path("/content/otco")
OUTPUT_DIR = Path("/content/otco_outputs") / EXPERIMENT_NAME
LOG_FILE = OUTPUT_DIR / "diagnostic_stdout.txt"


def require_a100():
    line = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
    name, total_mib = (part.strip() for part in line.split(","))
    print(f"Using GPU: {name} ({total_mib} MiB VRAM)")
    if "A100" not in name or int(total_mib) < 14000:
        raise RuntimeError("This frozen CLIP diagnostic requires an A100")


require_a100()
if PINNED_COMMIT == "__PIN_AFTER_REVIEW__":
    raise RuntimeError("Review and pin the diagnostic implementation before running")
token = userdata.get("GITHUB_TOKEN")
repo_url = (
    f"https://{token}@github.com/akashm776/otco.git"
    if token
    else "https://github.com/akashm776/otco.git"
)
if REPO_DIR.exists():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)
else:
    subprocess.run(
        ["git", "clone", "--no-checkout", repo_url, str(REPO_DIR)],
        check=True,
    )
subprocess.run(
    ["git", "-C", str(REPO_DIR), "checkout", "--detach", PINNED_COMMIT],
    check=True,
)
os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "datasets>=2.21.0,<3.0.0",
        "transformers==4.57.3",
        "pyyaml>=6.0.1",
        "pytest",
    ],
    check=True,
)
datasets = importlib.import_module("datasets")
print(f"datasets version: {datasets.__version__}")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Repository commit:", commit)
if commit != PINNED_COMMIT:
    raise RuntimeError(f"Expected pinned commit {PINNED_COMMIT}, got {commit}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "py_compile",
        "src/clip_negative_gradient_geometry.py",
        "src/clip_negative_gradient_geometry_metrics.py",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_clip_negative_gradient_geometry.py",
    ],
    check=True,
)
process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "src.clip_negative_gradient_geometry",
        "--config",
        CONFIG_FILE,
        "--output-directory",
        str(OUTPUT_DIR),
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
with LOG_FILE.open("w", encoding="utf-8") as log:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
return_code = process.wait()
if return_code:
    raise subprocess.CalledProcessError(return_code, process.args)

required = (
    "negative_gradient_geometry_report.json",
    "gradient_geometry_per_query.csv",
    "resolved_config.yaml",
    "diagnostic_holdout_indices.json",
    "diagnostic_stdout.txt",
)
for filename in required:
    if not (OUTPUT_DIR / filename).is_file():
        raise FileNotFoundError(f"Missing required diagnostic artifact: {filename}")
archive_path = Path("/content") / f"{EXPERIMENT_NAME}_results.zip"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(OUTPUT_DIR.parent))
files.download(str(archive_path))
