from google.colab import userdata, drive
import yaml, subprocess, sys, os

CONFIG_NAME = "hf_cub200_softmax_mix_adaptive"
CONFIG_FILE = f"configs/{CONFIG_NAME}.yaml"
RESULTS_FILE = f"results/cub200_{CONFIG_NAME}_results.txt"

token = userdata.get("GITHUB_TOKEN")
repo_url = f"https://{token}@github.com/akashm776/otco.git"
repo_dir = "/content/otco"
checkpoint_dir = "/content/drive/MyDrive/otco_checkpoints/cub200_softmax_mix_adaptive"

drive.mount('/content/drive')
os.makedirs(checkpoint_dir, exist_ok=True)

if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
else:
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

os.chdir(repo_dir)
subprocess.run([sys.executable, "-m", "pip", "install", "datasets<3.0.0", "pyyaml", "-q"], check=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.makedirs("results", exist_ok=True)

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
subprocess.run(["git", "-C", repo_dir, "commit", "-m", f"CUB-200 {CONFIG_NAME} run results"], check=False)
subprocess.run(["git", "-C", repo_dir, "push"], check=False)
