"""Paste into an A100 Colab cell, or use clip_gradient_stages.ipynb.

ARMS may be set in the calling notebook before executing this runner.
"""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from google.colab import files

PINNED_COMMIT = "f8b784becd5c074abc831f22ef17cb9fc4e9a88c"
ARMS = globals().get("ARMS", ["baseline", "uniform_top8", "hardest_real"])
ALLOWED_ARMS = {"baseline", "uniform_top8", "hardest_real", "pressure_matched"}
if not ARMS or len(set(ARMS)) != len(ARMS) or set(ARMS) - ALLOWED_ARMS:
    raise ValueError(f"Choose distinct arms from {sorted(ALLOWED_ARMS)}")

gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    text=True).splitlines()[0]
gpu_name, memory = [part.strip() for part in gpu.split(",")]
if "A100" not in gpu_name or int(memory) < 14000:
    raise RuntimeError(f"Select an A100 runtime for this controlled experiment; found {gpu}")
print("GPU:", gpu)

RUN_ID = "clip_gradient_stages_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
REPO_DIR = Path("/content") / f"otco_source_{RUN_ID}"
OUTPUT_DIR = Path("/content/otco_outputs") / RUN_ID
CHECKPOINT_DIR = Path("/content/otco_checkpoints") / RUN_ID
subprocess.run(["git", "clone", "--no-checkout", "https://github.com/akashm776/otco.git",
                str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--detach", PINNED_COMMIT], check=True)
os.chdir(REPO_DIR)
actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if actual_commit != PINNED_COMMIT:
    raise RuntimeError(f"Source pin mismatch: {actual_commit}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"
subprocess.run([sys.executable, "-m", "pip", "install", "datasets>=2.21.0,<3.0.0",
                "transformers==4.57.3", "pyyaml>=6.0.1", "pytest", "matplotlib"], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_clip_gradient_stages.py",
                "tests/test_clip_training.py", "tests/test_clip_negative_gradient_geometry.py",
                "tests/test_clip_negative_gradient_geometry_randomized.py"], check=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
manifest = {"run_id": RUN_ID, "source_commit": actual_commit, "arms": ARMS,
            "gpu": gpu, "python": sys.version, "completed_arms": [],
            "packages": {name: importlib.metadata.version(name) for name in
                         ["torch", "torchvision", "transformers", "datasets", "numpy",
                          "PyYAML", "matplotlib", "pytest"]}}
(OUTPUT_DIR / "protocol.yaml").write_bytes((REPO_DIR / "configs/clip_gradient_stages.yaml").read_bytes())


def save_manifest():
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def download_bundle(label):
    archive_path = Path("/content") / f"{RUN_ID}_{label}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(OUTPUT_DIR.parent))
    print(f"Results: {archive_path} ({archive_path.stat().st_size / 2**20:.1f} MiB)")
    try:
        files.download(str(archive_path))
    except Exception as error:
        print(f"Automatic download failed ({type(error).__name__}); download the ZIP from Colab's Files pane.")


save_manifest()
try:
    for arm in ARMS:
        manifest["active_arm"] = arm
        save_manifest()
        command = [sys.executable, "-u", "-m", "src.clip_gradient_stages", "--arm", arm,
                   "--output-directory", str(OUTPUT_DIR),
                   "--checkpoint-directory", str(CHECKPOINT_DIR)]
        print(f"Starting {arm}; four fixed-batch diagnostics during 50 training epochs", flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            with (OUTPUT_DIR / f"{arm}_stdout.txt").open("w") as log:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        reports = json.loads((OUTPUT_DIR / arm / "diagnostics/stage_summary.json").read_text())
        if [report["completed_updates"] for report in reports] != [0, 1001, 2001, 3850]:
            raise RuntimeError(f"{arm}: missing or incorrect stage measurements")
        if not (OUTPUT_DIR / arm / "training/summary.json").is_file():
            raise RuntimeError(f"{arm}: missing training completion summary")
        manifest["completed_arms"].append(arm)
        save_manifest()
        download_bundle(f"through_{arm}")
    subprocess.run([sys.executable, "-m", "src.clip_gradient_stages", "--plot-only",
                    "--output-directory", str(OUTPUT_DIR)], check=True)
    manifest["status"] = "complete"
except BaseException as error:
    manifest["status"] = "interrupted_or_failed"
    manifest["error"] = f"{type(error).__name__}: {error}"
    raise
finally:
    save_manifest()
    download_bundle(manifest.get("status", "partial"))

print("Finished. Reports, per-query CSVs, features and trend plots are in the final ZIP.")
print("Encoder/optimizer checkpoints (not included in ZIP):", CHECKPOINT_DIR)
