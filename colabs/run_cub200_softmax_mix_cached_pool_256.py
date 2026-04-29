from google.colab import userdata, drive
import yaml, subprocess, sys, os

# ── Experiment identity ────────────────────────────────────────────────────────
CONFIG_NAME   = "hf_cub200_softmax_mix_cached_pool_256"
CONFIG_FILE   = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE  = "results/CUB_200_OT-MIX_cached_pool_256.txt"
BRANCH        = "main"

# ── Paths ──────────────────────────────────────────────────────────────────────
token         = userdata.get("GITHUB_TOKEN")
repo_url      = f"https://{token}@github.com/akashm776/otco.git"
repo_dir      = "/content/otco"
checkpoint_dir = "/content/drive/MyDrive/otco_checkpoints/cub200_softmax_mix_cached_pool_256"

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
# Phase III v3: detached cached-pool OT-Mix with pool_size=256.
# Identical to the record-holding cached_pool_128 run (best Avg R@1=1.48% at ep49)
# except pool_size doubled to 256.
#
# Hypothesis: larger random support gives OT more candidates to find hard
# negatives, improving plan quality and T→I retrieval.
#
# Key diagnostics:
#   pool_selected_rank_mean   — expect lower than N=128 run if harder negs found
#   coupling_entropy          — expect similar 2.9–3.1 range
#   T→I R@1, I→T R@1         — compare to v1: 1.36% / 1.61% / Avg 1.48%
#
# Note: pool embed rebuild is ~20 forward passes at 256-batch → still negligible.
# No gradient through pool embeddings — detached.

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
     "CUB-200 cached_pool_256 run results (Phase III v3)"],
    check=False,
)
subprocess.run(["git", "-C", repo_dir, "push"], check=False)
