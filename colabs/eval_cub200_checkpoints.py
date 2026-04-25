from google.colab import userdata, drive
import yaml, subprocess, sys, os, torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "/content/drive/MyDrive/otco_checkpoints/cub200_baseline"
EPOCHS_TO_EVAL = [5, 10, 15]
RUN_CONFIG = "configs/hf_cub200_baseline.yaml"
# ─────────────────────────────────────────────────────────────────────────────

token = userdata.get("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/akashm776/otco.git"
repo_dir = "/content/otco"

drive.mount('/content/drive')

if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
else:
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

os.chdir(repo_dir)
subprocess.run([sys.executable, "-m", "pip", "install", "datasets<3.0.0", "pyyaml", "-q"], check=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── imports after chdir ───────────────────────────────────────────────────────
from transformers import AutoModel
from src.config_loader import load_run_config
from src.data_setup import build_data_bundle
from model.model import OTLIP
from src.utils import evaluate_retrieval, evaluate_image_to_text_retrieval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── load config + val data ────────────────────────────────────────────────────
with open(RUN_CONFIG) as f:
    raw = yaml.safe_load(f)
raw["experiment"]["overrides"]["num_workers"] = 2
with open(RUN_CONFIG, "w") as f:
    yaml.dump(raw, f)
run = load_run_config(RUN_CONFIG)
config = run["experiment_config"]
dataset_cfg = run["dataset"]

data = build_data_bundle(config=config, root_dir=os.getcwd(), dataset_cfg=dataset_cfg)
val_loader_canonical = data.val_loader_canonical
print(f"Val loader ready: {len(val_loader_canonical.dataset)} samples")

# ── build model skeleton (weights loaded per checkpoint) ─────────────────────
vision_model = AutoModel.from_pretrained(config.get("model_vision", "microsoft/resnet-50"))
text_model   = AutoModel.from_pretrained(config.get("model_text", "distilbert-base-uncased"))
model = OTLIP(vision_model=vision_model, text_model=text_model).to(device)

# ── eval loop ─────────────────────────────────────────────────────────────────
results = []

for epoch in EPOCHS_TO_EVAL:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
    if not os.path.exists(ckpt_path):
        print(f"\nEpoch {epoch}: checkpoint not found at {ckpt_path}, skipping")
        continue

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion_step = ckpt.get("criterion_step", "?")
    saved_epoch    = ckpt.get("epoch", "?")

    t2i = evaluate_retrieval(model, val_loader_canonical, device)
    i2t = evaluate_image_to_text_retrieval(model, val_loader_canonical, device)

    avg_r1 = (t2i["R@1"] + i2t["R@1"]) / 2.0
    results.append({
        "epoch": epoch,
        "criterion_step": criterion_step,
        "t2i_r1": t2i["R@1"],
        "i2t_r1": i2t["R@1"],
        "avg_r1": avg_r1,
    })

    print(f"\nEpoch {epoch:>2}  criterion_step={criterion_step:>6}"
          f"  T→I={t2i['R@1']:.2f}%  I→T={i2t['R@1']:.2f}%  Avg={avg_r1:.2f}%")

# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Epoch':>6} {'Criterion Step':>15} {'T→I R@1':>10} {'I→T R@1':>10} {'Avg R@1':>10}")
print("-"*65)
for r in results:
    print(f"{r['epoch']:>6} {str(r['criterion_step']):>15} {r['t2i_r1']:>9.2f}% {r['i2t_r1']:>9.2f}% {r['avg_r1']:>9.2f}%")
print("="*65)
