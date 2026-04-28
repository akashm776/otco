from google.colab import userdata, drive
import yaml, subprocess, sys, os

# ── Experiment identity ────────────────────────────────────────────────────────
CONFIG_NAME   = "hf_cub200_softmax_mix_cached_pool_128"
CONFIG_FILE   = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE  = "results/CUB_200_OT-MIX_cached_pool_128.txt"
BRANCH        = "claude/jovial-beaver-13d765"   # change to "main" once merged

# ── Paths ──────────────────────────────────────────────────────────────────────
token         = userdata.get("GITHUB_TOKEN")
repo_url      = f"https://{token}@github.com/akashm776/otco.git"
repo_dir      = "/content/otco"
checkpoint_dir = "/content/drive/MyDrive/otco_checkpoints/cub200_softmax_mix_cached_pool_128"

# ── Setup ──────────────────────────────────────────────────────────────────────
drive.mount('/content/drive')
os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", repo_dir, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", repo_dir, "pull", "origin", BRANCH], check=True)
else:
    subprocess.run(["git", "clone", "--branch", BRANCH, repo_url, repo_dir], check=True)

os.chdir(repo_dir)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "datasets<3.0.0", "pyyaml", "-q"],
    check=True,
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("results", exist_ok=True)

# ── Patch num_workers for Colab (2 CPUs available) ────────────────────────────
# num_workers=2 applies to both the training DataLoader and the pool DataLoader
# built inside _build_image_pool at each epoch start.
with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["overrides"]["num_workers"] = 2
with open(CONFIG_FILE, "w") as f:
    yaml.dump(cfg, f)

# ── Train ──────────────────────────────────────────────────────────────────────
# Phase III / Option A: epoch-start image embedding cache (pool_size=128).
# Each epoch:
#   1. Forward all ~5000 training images (no_grad, eval transform) → cache [5000, 512]
#   2. Per step: sample 128 from cache excluding current batch positives
#   3. OT over [B=64, N=128] → detached barycentric synthetic negative
#   4. Adaptive gated OT loss (same gating as cub200_softmax_mix_adaptive_gated)
#
# Key diagnostics to watch:
#   pool_selected_rank_mean   — expect <<64 if OT finds hard negatives from pool
#   coupling_entropy          — expect 2.0–2.5 for sharp plans
#   pos_selected_gap          — expect near boundary (–0.05 to 0.00)
#   gap_bucket_useful count   — how often OT is active and near-boundary
#   T→I R@1, I→T R@1         — compare to baseline 1.05 / 1.71
#
# Note: pool embedding rebuild adds ~20 forward batches per epoch (negligible).
# No gradient flows through pool embeddings — text encoder trains via synthetic
# loss; image encoder trains only via base SigLIP loss.

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
     "CUB-200 cached_pool_128 run results (Phase III v1)"],
    check=False,
)
subprocess.run(["git", "-C", repo_dir, "push"], check=False)
