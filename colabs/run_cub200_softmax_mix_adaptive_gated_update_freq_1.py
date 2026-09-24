"""Run the fresh-plan adaptive-gated CUB-200 control in Google Colab."""

import importlib
import glob
import os
import shutil
import subprocess
import sys
import zipfile

import yaml
from google.colab import files, userdata


CONFIG_NAME = "hf_cub200_softmax_mix_adaptive_gated_update_freq_1"
CONFIG_FILE = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE = "results/CUB_200_OT-MIX_adaptive_gated_update_freq_1.txt"
EXPERIMENT_NAME = "cub200_softmax_mix_adaptive_gated_update_freq_1"
BASE_EXPERIMENT_NAME = "cub200_softmax_mix_adaptive_gated"


def require_adequate_gpu(min_vram_mib=14000):
    """Refuse to start an expensive training run without a usable Colab GPU."""
    try:
        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "No NVIDIA GPU is attached. In Colab, select Runtime > Change "
            "runtime type > Hardware accelerator > GPU, then run again."
        ) from error

    gpu_name, gpu_vram = (part.strip() for part in gpu_info.splitlines()[0].split(","))
    gpu_vram_mib = int(gpu_vram)
    print(f"Using GPU: {gpu_name} ({gpu_vram_mib} MiB VRAM)")
    if gpu_vram_mib < min_vram_mib:
        raise RuntimeError(
            f"The attached {gpu_name} has only {gpu_vram_mib} MiB VRAM; "
            f"this batch-64 run requires at least {min_vram_mib} MiB. "
            "Reconnect with a T4, L4, A100, or better GPU."
        )


require_adequate_gpu()

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError(
        "Missing Colab secret GITHUB_TOKEN. Add a GitHub token in the Colab "
        "Secrets panel and grant this notebook access to it."
    )
repo_url = f"https://{token}@github.com/akashm776/otco.git"
repo_dir = "/content/otco"
checkpoint_dir = (
    "/content/otco_checkpoints/"
    "cub200_softmax_mix_adaptive_gated_update_freq_1"
)

# Keep checkpoints on the Colab VM rather than consuming Google Drive space.
# Colab local storage is ephemeral, so completed metrics are also committed to
# the repository below before the runtime is disconnected.
os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
else:
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

os.chdir(repo_dir)
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "datasets"],
    check=False,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "datasets<3.0.0",
        "pyyaml",
    ],
    check=True,
)
shutil.rmtree(
    os.path.expanduser("~/.cache/huggingface/datasets"),
    ignore_errors=True,
)
datasets = importlib.import_module("datasets")
print(f"datasets version: {datasets.__version__}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs("results", exist_ok=True)
existing_experiment_logs = set(glob.glob("experiments/exp_*.json"))

# Materialize the unpublished control from the adaptive-gated experiment that
# already exists on GitHub. This makes the runner usable from an older notebook
# without first pushing the new experiment files.
with open("configs/experiments.yaml", encoding="utf-8") as config_stream:
    experiments_config = yaml.safe_load(config_stream)
base_experiment = experiments_config["experiments"][BASE_EXPERIMENT_NAME]
experiments_config["experiments"][EXPERIMENT_NAME] = {
    **base_experiment,
    "update_freq": 1,
}
with open("configs/experiments.yaml", "w", encoding="utf-8") as config_stream:
    yaml.safe_dump(experiments_config, config_stream, sort_keys=False)

# Colab performs more reliably with two data-loading worker processes.
config = {
    "experiment": {
        "name": EXPERIMENT_NAME,
        "experiments_file": "experiments.yaml",
        "overrides": {"num_workers": 2},
    },
    "dataset": {
        "backend": "hf_cub200",
        "local": {"root_dir": None},
        "hf": {
            "dataset_name": "alkzar90/CC6204-Hackaton-Cub-Dataset",
            "train_split": "train",
            "val_split": "test",
        },
    },
    "model": {"vision": None, "text": None},
    "run": {"checkpoint_dir": None, "print_experiment_comparison": True},
}
with open(CONFIG_FILE, "w", encoding="utf-8") as config_stream:
    yaml.safe_dump(config, config_stream, sort_keys=False)

process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "src.main",
        "--config",
        CONFIG_FILE,
        "--checkpoint-dir",
        checkpoint_dir,
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
with open(RESULTS_FILE, "w", encoding="utf-8") as log:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
return_code = process.wait()
if return_code != 0:
    raise subprocess.CalledProcessError(return_code, process.args)

# Preserve the small, readable artifacts both on GitHub and through a browser
# download. Model checkpoints remain in /content because they are much larger.
new_experiment_logs = sorted(
    set(glob.glob("experiments/exp_*.json")) - existing_experiment_logs
)
artifact_archive = "/content/cub200_adaptive_gated_update_freq_1_results.zip"
with zipfile.ZipFile(artifact_archive, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(RESULTS_FILE, arcname=os.path.basename(RESULTS_FILE))
    archive.write(CONFIG_FILE, arcname=os.path.basename(CONFIG_FILE))
    for experiment_log in new_experiment_logs:
        archive.write(
            experiment_log,
            arcname=os.path.join("experiments", os.path.basename(experiment_log)),
        )

# Download first, so a later authentication or push failure cannot prevent the
# browser from receiving a durable copy of the completed results.
files.download(artifact_archive)

# Preserve the two useful training states without duplicating them into an
# archive (which would temporarily consume roughly twice the checkpoint space).
# best_model.pt is the highest-validation checkpoint; latest.pt is the final
# epoch checkpoint and can be used to resume training.
for checkpoint_name in ("best_model.pt", "latest.pt"):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print(f"Downloading {checkpoint_name} to the browser computer...")
        files.download(checkpoint_path)
    else:
        print(f"Warning: expected checkpoint was not found: {checkpoint_path}")

# Use the repository owner's identity for the completed-run commit.
subprocess.run(
    ["git", "-C", repo_dir, "config", "user.name", "Akash Mittal"],
    check=True,
)
subprocess.run(
    ["git", "-C", repo_dir, "config", "user.email", "akashmit28@gmail.com"],
    check=True,
)
subprocess.run(
    ["git", "-C", repo_dir, "add", RESULTS_FILE, *new_experiment_logs],
    check=True,
)
subprocess.run(
    [
        "git",
        "-C",
        repo_dir,
        "commit",
        "-m",
        "Add CUB-200 fresh-plan adaptive-gated results",
    ],
    check=True,
)
subprocess.run(["git", "-C", repo_dir, "push", "origin", "HEAD:main"], check=True)
print("Results committed and pushed to GitHub successfully.")
