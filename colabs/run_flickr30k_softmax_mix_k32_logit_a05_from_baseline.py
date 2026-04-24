from google.colab import userdata, drive
import yaml, subprocess, sys, os, torch

CONFIG_NAME = "hf_flickr30k_softmax_mix_k32_logit_a05_from_baseline"
CONFIG_FILE = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE = f"results/flickr30k_{CONFIG_NAME}_results.txt"

token = userdata.get("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/akashm776/otco.git"
repo_dir = "/content/otco"
checkpoint_dir = "/content/drive/MyDrive/otco_checkpoints/softmax_mix_k32_logit_a05_from_baseline"
baseline_best  = "/content/drive/MyDrive/otco_checkpoints/best_model.pt"

drive.mount('/content/drive')
os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
else:
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

os.chdir(repo_dir)
subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "pyyaml", "-q"], check=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs("results", exist_ok=True)

# Bootstrap latest.pt from baseline best_model.pt if no checkpoint exists yet
latest_path = os.path.join(checkpoint_dir, "latest.pt")
if not os.path.exists(latest_path):
    if os.path.exists(baseline_best):
        print(f"Bootstrapping from baseline best model: {baseline_best}")
        ckpt = torch.load(baseline_best, map_location="cpu")
        torch.save({
            "epoch": -1,           # main.py sets start_epoch = epoch + 1 = 0
            "global_step": 0,
            "model_state_dict": ckpt["model_state_dict"],
            "optimizer_state_dict": ckpt["optimizer_state_dict"],
            "loss": ckpt.get("loss", 0.0),
            "best_avg_recall": 0.0,
            "criterion_step": 0,   # OT-Mix step counter starts fresh
        }, latest_path)
        print("Bootstrap complete — alpha=0.05 OT-Mix starts from baseline weights, epoch 0")
    else:
        print(f"WARNING: {baseline_best} not found — starting from scratch")
else:
    print(f"Found existing checkpoint — resuming from: {latest_path}")

with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["overrides"]["num_workers"] = 2
with open(CONFIG_FILE, "w") as f:
    yaml.dump(cfg, f)

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

subprocess.run(["git", "-C", repo_dir, "add", "experiments/", "results/"], check=False)
subprocess.run(["git", "-C", repo_dir, "commit", "-m", f"Flickr30k {CONFIG_NAME} run results"], check=False)
subprocess.run(["git", "-C", repo_dir, "push"], check=False)
