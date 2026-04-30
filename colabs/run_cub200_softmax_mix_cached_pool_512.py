from google.colab import userdata, drive
import yaml, subprocess, sys, os

# ── Experiment identity ────────────────────────────────────────────────────────
CONFIG_NAME   = "hf_cub200_softmax_mix_cached_pool_512"
CONFIG_FILE   = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE  = "results/CUB_200_OT-MIX_cached_pool_512.txt"
BRANCH        = "main"

# ── Paths ──────────────────────────────────────────────────────────────────────
token         = userdata.get("GITHUB_TOKEN")
repo_url      = f"https://{token}@github.com/akashm776/otco.git"
repo_dir      = "/content/otco"
checkpoint_dir = "/content/drive/MyDrive/otco_checkpoints/cub200_softmax_mix_cached_pool_512"

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
# Phase III v3 saturation check: detached cached-pool OT-Mix with pool_size=512.
# Identical to cached_pool_256 (best Avg R@1=1.49% at ep47) except pool doubled.
#
# Hypothesis: if N=256 introduces more too_easy suppression than N=128, N=512
# should show even more — either confirming diminishing returns (saturation) or
# revealing a quality threshold where random sampling breaks down.
#
# Key diagnostics vs N=128/256:
#   gap_bucket_too_easy count   — expect higher than N=256
#   pool_selected_rank_mean     — expect lower (harder negatives in larger pool)
#   coupling_entropy            — expect similar 2.9–3.1 range
#   T→I R@1, I→T R@1           — compare to N=128: 1.36%/1.61%, N=256: 1.26%/1.73%
#
# Note: pool embed rebuild unchanged (~20 forward passes). Detached.

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
     "CUB-200 cached_pool_512 run results (Phase III v3 saturation check)"],
    check=False,
)
subprocess.run(["git", "-C", repo_dir, "push"], check=False)
