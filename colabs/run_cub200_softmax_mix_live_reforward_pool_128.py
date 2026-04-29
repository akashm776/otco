from google.colab import userdata, drive
import yaml, subprocess, sys, os

# ── Experiment identity ────────────────────────────────────────────────────────
CONFIG_NAME   = "hf_cub200_softmax_mix_live_reforward_pool_128"
CONFIG_FILE   = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE  = "results/CUB_200_OT-MIX_live_reforward_pool_128.txt"
BRANCH        = "main"

# ── Paths ──────────────────────────────────────────────────────────────────────
token         = userdata.get("GITHUB_TOKEN")
repo_url      = f"https://{token}@github.com/akashm776/otco.git"
repo_dir      = "/content/otco"
checkpoint_dir = "/content/drive/MyDrive/otco_checkpoints/cub200_softmax_mix_live_reforward_pool_128"

# ── Setup ──────────────────────────────────────────────────────────────────────
if not os.path.isdir('/content/drive/MyDrive'):
    drive.mount('/content/drive')
os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", repo_dir, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", repo_dir, "pull", "origin", BRANCH], check=True)
else:
    subprocess.run(["git", "clone", "--branch", BRANCH, repo_url, repo_dir], check=True)

os.chdir(repo_dir)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "datasets"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "datasets<3.0.0", "pyyaml"], check=True)
import shutil; shutil.rmtree(os.path.expanduser("~/.cache/huggingface/datasets"), ignore_errors=True)
import importlib, datasets as _ds; print(f"datasets version: {_ds.__version__}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("results", exist_ok=True)

# ── Patch num_workers for Colab (2 CPUs available) ────────────────────────────
with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["overrides"]["num_workers"] = 2
with open(CONFIG_FILE, "w") as f:
    yaml.dump(cfg, f)

# ── Train ──────────────────────────────────────────────────────────────────────
# Phase III v2: cached-pool OT discovery + live re-forward of top contributors.
# Each epoch:
#   1. Forward all ~5994 training images (no_grad, eval transform) → cache [5994, 512]
#   2. Per step: sample 128 from cache excluding current batch positives
#   3. Compute OT plan (no_grad, detached) over [B=64, N=128]
#   4. Select top-2 contributors per query → deduplicate → cap at 64 unique images
#   5. Re-forward selected images through live image encoder (with gradient)
#   6. Build barycentric synthetic from live features → gradient flows to image encoder
#   7. Adaptive gated OT loss (same gating as cub200_softmax_mix_adaptive_gated)
#
# Key diagnostics:
#   Live Re-forward:   num unique images re-forwarded per step (target ~30-50)
#   Mass Retained:     fraction of OT plan mass in live set (target >0.90)
#   pool_selected_rank_mean   — expect <<64 if OT finds hard negatives from pool
#   coupling_entropy          — expect 2.0–3.0 for useful plans
#   T→I R@1, I→T R@1         — compare to baseline 1.05 / 1.71, v1 1.36 / 1.61
#
# Expected: T→I ~1.30-1.40%, I→T 1.70%+, Avg ~1.50%+
# Asymmetry fix vs v1: live re-forward restores gradient to image encoder.

process = subprocess.Popen(
    [sys.executable, "-m", "src.main",
     "--config", CONFIG_FILE,
     "--checkpoint-dir", checkpoint_dir],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
with open(RESULTS_FILE, "w") as log:
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
process.wait()

# ── Commit results ─────────────────────────────────────────────────────────────
subprocess.run(["git", "-C", repo_dir, "add", "experiments/", "results/"], check=False)
subprocess.run(
    ["git", "-C", repo_dir, "commit", "-m",
     "CUB-200 live_reforward_pool_128 run results (Phase III v2)"],
    check=False,
)
subprocess.run(["git", "-C", repo_dir, "push"], check=False)
