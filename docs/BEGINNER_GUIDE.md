# Beginner's End-to-End Alpamayo 1.5 Reproduction Guide

This guide explains the complete path from an empty remote server to a working Alpamayo 1.5 video demonstration. It is written for readers who have never deployed a large CUDA model, authenticated with Hugging Face, or moved a model cache between servers.

The sequence is based on the actual reproduction behind this repository. The first machine used an RTX 4060 Ti with 8 GB of VRAM. It was sufficient for cloning, environment setup, authorization, and downloading assets, but not for model inference. The completed inference was moved to an NVIDIA A100 80 GB server.

## 1. Final Result

At the end, you will have:

- The official `NVlabs/alpamayo1.5` source code
- A Python 3.12 virtual environment managed with `uv`
- PyTorch 2.8 with CUDA support
- Authorized access to the NVIDIA model and dataset
- Approximately 22 GB of Alpamayo model weights in the Hugging Face cache
- A successful official-style inference using PyTorch SDPA
- Five sliding-window predictions over each 10-second clip
- Front-camera videos containing predictions, ground truth, and CoC text
- A synchronized 2x2 composite MP4

## 2. Concepts You Need to Know

### SSH

SSH opens a terminal on another computer. Commands entered after SSH login run on the server, not on your Windows computer.

### GPU memory

GPU memory, or VRAM, is different from normal RAM. Alpamayo single-sample inference needs approximately 24 GB of VRAM. Having a large amount of system RAM does not make an 8 GB GPU sufficient.

### Virtual environment

A virtual environment keeps this project's Python packages separate from system Python. This guide names it `a1_5_venv`.

### Hugging Face cache

Downloaded files are normally stored under:

```text
~/.cache/huggingface/hub/
```

This cache can be copied to another server so that the 22 GB checkpoint does not need to be downloaded twice.

### Gated resource

A gated model or dataset requires you to accept terms in a browser. A valid token is not enough if the account has not accepted the gated page.

## 3. Hardware and Storage Checklist

Use a server with:

- Linux, preferably Ubuntu
- NVIDIA GPU with CUDA support
- At least 24 GB VRAM for one trajectory sample
- Python 3.12, or permission for `uv` to install it
- At least 80 GB of free disk space
- Stable GitHub and Hugging Face access, or a trusted mirror

Official approximate VRAM requirements are:

| Inference mode | Approximate VRAM |
|---|---:|
| One trajectory sample | 24 GB |
| 16 trajectory samples | 40 GB |
| 16 samples with classifier-free guidance | 60 GB |

An RTX 3090, A100, H100, or B200 is appropriate for single-sample inference. An 8 GB RTX 4060 Ti is not.

## 4. Request Every Required Hugging Face Access

Create a Hugging Face account and, while logged in to the same account, accept the terms on all three pages:

1. [PhysicalAI Autonomous Vehicles Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
2. [Alpamayo-1.5-10B](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
3. [Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B)

The third page matters. During the actual reproduction, dataset access succeeded but inference later stopped because the internal Cosmos dependency was still gated.

Create a **read-only** token at [Hugging Face settings](https://huggingface.co/settings/tokens). Do not put the token in a chat, README, issue, script, or Git commit.

## 5. Connect from Windows

Open PowerShell or Windows Terminal. Replace the placeholders:

```powershell
ssh <SERVER_USER>@<SERVER_IP>
```

On first connection, verify the host fingerprint with the server administrator before accepting it. After login, confirm the machine:

```bash
whoami
hostname
pwd
```

All following Linux commands run inside SSH unless a section explicitly says to use Windows.

## 6. Inspect the Server

Check the GPU and current processes:

```bash
nvidia-smi
```

Check CUDA, disk, and basic tools:

```bash
nvcc --version
df -h "$HOME"
git --version
curl --version
python3 --version
```

The SDPA path does not require compiling FlashAttention, but `nvcc` is required if you choose the default FlashAttention path. Do not continue if disk space is nearly exhausted.

## 7. Clone the Official Repository

```bash
cd "$HOME"
git clone https://github.com/NVlabs/alpamayo1.5.git
cd alpamayo1.5
```

Confirm the important files:

```bash
test -f pyproject.toml && echo "pyproject.toml found"
test -f src/alpamayo1_5/test_inference.py && echo "test script found"
```

## 8. Install uv

Use the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If that download stalls, use PyPI:

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If the default PyPI route is slow, use a trusted regional mirror. For example:

```bash
python3 -m pip install --user \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  uv
```

## 9. Create the Python 3.12 Environment

The upstream project requires Python `3.12.x` exactly:

```bash
uv python install 3.12
cd "$HOME/alpamayo1.5"
uv venv a1_5_venv --python 3.12
source a1_5_venv/bin/activate
python --version
```

The last command must print Python 3.12.

## 10. Install Dependencies

The tested reproduction skipped FlashAttention and used PyTorch SDPA:

```bash
uv sync --active --no-install-package flash-attn
```

This downloads PyTorch and CUDA runtime packages and can take several minutes. A dependency check may report that `flash-attn` is absent; that is expected for this path.

If CUDA Toolkit 12.x and `nvcc` are correctly installed, you may instead try the upstream default:

```bash
uv sync --active
```

If it fails while compiling `flash-attn`, return to the SDPA command.

## 11. Verify Python, CUDA, and Imports

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.tensor([1.0, 2.0], device="cuda")
    print("CUDA calculation:", (x * 2).tolist())
PY
```

The tested environment reported `torch 2.8.0+cu128` and a successful CUDA calculation.

Verify the project:

```bash
python - <<'PY'
import alpamayo1_5
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

print("Project imports are working.")
PY
```

Do not start a 22 GB download until both checks pass.

## 12. Log In to Hugging Face Safely

```bash
cd "$HOME/alpamayo1.5"
source a1_5_venv/bin/activate
hf auth login
hf auth whoami
```

Paste the read-only token only at the hidden prompt. Avoid putting it after `--token`, because the command may be stored in shell history.

If direct requests time out and a trusted compatible mirror is available:

```bash
export HF_ENDPOINT=https://hf-mirror.com
HF_ENDPOINT=https://hf-mirror.com hf auth whoami
```

## 13. Test Authorization with a Small File

This downloads only `config.json`, not the full checkpoint:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="nvidia/Alpamayo-1.5-10B",
    filename="config.json",
)
print("Downloaded:", path)
PY
```

This verifies model access, but dataset access is checked only when the example clip is loaded.

## 14. Run the Official Demo

```bash
cd "$HOME/alpamayo1.5"
source a1_5_venv/bin/activate
export PYTHONUNBUFFERED=1
python src/alpamayo1_5/test_inference.py 2>&1 | tee demo_run.log
```

The first successful run downloads a small example clip and approximately 22 GB of model weights. Use `tmux` for long downloads:

```bash
tmux new -s alpamayo
```

Run the demo inside `tmux`. Detach with `Ctrl+B`, then `D`. Reconnect with:

```bash
tmux attach -t alpamayo
```

## 15. Fix Gated-Access Errors

### Dataset 403

Actual error:

```text
GatedRepoError: 403 Client Error
Cannot access gated repo: nvidia/PhysicalAI-Autonomous-Vehicles
```

Fix:

1. Accept dataset access in the browser.
2. Confirm `hf auth whoami` shows the same account.
3. Log in again if the server cached a different account.

### Cosmos access

The next actual blocker was:

```text
Cannot access gated repo: nvidia/Cosmos-Reason2-8B
```

Accept the Cosmos terms with the same account and rerun. Existing dataset downloads are reused.

## 16. Use the SDPA Fallback

When `flash-attn` is not installed, load the model this way:

```python
model = Alpamayo1_5.from_pretrained(
    "nvidia/Alpamayo-1.5-10B",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda")
```

The scripts in this repository already use this setting, so you do not need to edit the upstream test file when using them.

## 17. Recognize a Real CUDA OOM

The RTX 4060 Ti reached checkpoint loading and then failed at `.to("cuda")`:

```text
torch.OutOfMemoryError: CUDA out of memory
GPU total capacity: 7.61 GiB
PyTorch allocated: approximately 7.34 GiB
Tried to allocate an additional 32 MiB
```

This is not an authorization or package problem. Re-running cannot make an 8 GB GPU meet a 24 GB requirement. Move to a larger GPU.

## 18. Prepare the A100

Connect to the second server:

```powershell
ssh <A100_USER>@<A100_IP>
```

Repeat the server inspection, repository clone, `uv` installation, Python 3.12 environment, SDPA dependency installation, and Hugging Face login.

Do **not** copy the old virtual environment. It contains absolute paths and packages tied to the first server. Recreate it cleanly on the A100.

## 19. Download Directly or Copy the Cache

If the A100 has fast Hugging Face access, run the demo and let Hub populate its cache. To prefetch the main checkpoint:

```bash
hf download nvidia/Alpamayo-1.5-10B
```

If the A100 network is slow, copy the existing cache from the first server.

### Create a temporary key on the source server

```bash
ssh-keygen \
  -t ed25519 \
  -f "$HOME/.ssh/alpamayo_transfer" \
  -N "" \
  -C "alpamayo-transfer-temporary"

ssh-copy-id \
  -i "$HOME/.ssh/alpamayo_transfer.pub" \
  <A100_USER>@<A100_IP>
```

Test it:

```bash
ssh -i "$HOME/.ssh/alpamayo_transfer" <A100_USER>@<A100_IP> hostname
```

Create the destination and copy selected caches:

```bash
ssh -i "$HOME/.ssh/alpamayo_transfer" <A100_USER>@<A100_IP> \
  'mkdir -p ~/.cache/huggingface/hub'

for cache_dir in \
  models--nvidia--Alpamayo-1.5-10B \
  models--nvidia--Cosmos-Reason2-8B \
  datasets--nvidia--PhysicalAI-Autonomous-Vehicles
do
  if [ -d "$HOME/.cache/huggingface/hub/$cache_dir" ]; then
    rsync -avhP \
      -e "ssh -i $HOME/.ssh/alpamayo_transfer" \
      "$HOME/.cache/huggingface/hub/$cache_dir/" \
      "<A100_USER>@<A100_IP>:.cache/huggingface/hub/$cache_dir/"
  fi
done
```

`rsync -P` resumes partial transfers. Do not copy the complete cache if it contains unrelated large models.

Verify on the A100:

```bash
du -sh ~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B
du -sh ~/.cache/huggingface/hub/models--nvidia--Cosmos-Reason2-8B
du -sh ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles
```

The Alpamayo directory should be approximately 21-22 GB.

Remove the temporary key on the source:

```bash
rm -f "$HOME/.ssh/alpamayo_transfer" "$HOME/.ssh/alpamayo_transfer.pub"
```

Remove only its matching line on the A100:

```bash
sed -i '/alpamayo-transfer-temporary/d' "$HOME/.ssh/authorized_keys"
```

Never delete the complete `authorized_keys` file.

## 20. Check A100 Availability

```bash
nvidia-smi
```

In the tested run, another process used approximately 44 GB of the 80 GB A100. Alpamayo added approximately 21.8 GB. The scripts use these guards:

```text
Before model loading: at least 30,000 MiB free
Before every window: at least 8,000 MiB free
```

Do not stop another user's process. Wait or use another GPU when the guard fails.

## 21. Install the Video Scripts

On the A100:

```bash
cd "$HOME"
git clone https://github.com/130070/Alpamayo1.5-VLA.git
cp "$HOME/Alpamayo1.5-VLA/scripts/"*.py "$HOME/alpamayo1.5/"
cd "$HOME/alpamayo1.5"
```

The scripts import the upstream package, so run them from the upstream repository environment.

## 22. Run One 10-Second Scene

```bash
cd "$HOME/alpamayo1.5"
source a1_5_venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com  # Remove if direct access works.
export PYTHONUNBUFFERED=1
python alpamayo_10s_sliding_demo.py 2>&1 | tee outputs/alpamayo_10s_run.log
```

The script performs five real inferences at two-second intervals. It writes a `.pt` cache after each window, so rerunning resumes missing windows.

Expected files:

```text
outputs/alpamayo_10s/
|-- alpamayo_10s_input.mp4
|-- alpamayo_10s_result.mp4
|-- alpamayo_10s_result_poster.png
|-- alpamayo_10s_meta.json
`-- alpamayo_10s_sliding_results.pt
```

## 23. Build Three Scenes and the 2x2 Video

```bash
python alpamayo_three_scenes_and_grid.py 2>&1 | tee outputs/alpamayo_batch_run.log
```

The batch script loads the model once, processes all missing windows, releases GPU memory, renders on CPU, and uses FFmpeg to create:

```text
outputs/alpamayo_2x2/alpamayo_2x2_10s.mp4
```

## 24. Verify the MP4

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
  -of json \
  outputs/alpamayo_2x2/alpamayo_2x2_10s.mp4
```

Expected values:

```text
H.264, 1920x1080, 5 FPS, 10 seconds, 50 frames
```

## 25. Download Results to Windows

Open PowerShell on Windows:

```powershell
New-Item -ItemType Directory -Force .\alpamayo-results

scp <A100_USER>@<A100_IP>:~/alpamayo1.5/outputs/alpamayo_2x2/alpamayo_2x2_10s.mp4 `
  .\alpamayo-results\

scp -r <A100_USER>@<A100_IP>:~/alpamayo1.5/outputs/alpamayo_10s `
  .\alpamayo-results\
```

Use an SFTP client or `rsync` from WSL if resume support is required.

## 26. Common Problems

| Symptom | Meaning | Fix |
|---|---|---|
| Dataset `403 GatedRepoError` | Dataset access not accepted | Accept terms and verify `hf auth whoami` |
| Cosmos gated error | Internal dependency not authorized | Accept the Cosmos model terms |
| Hugging Face timeout | Network route is slow or blocked | Check DNS/proxy or use a trusted mirror |
| `flash-attn` build fails | CUDA compiler path is incomplete | Skip it and use SDPA |
| OOM on 8/16 GB GPU | GPU is below requirement | Use a 24 GB or larger GPU |
| Process exits with SSH | It was tied to the terminal | Use `tmux` or correctly configured `nohup` |
| Empty log | Python output was buffered | Set `PYTHONUNBUFFERED=1` |
| Overlay misses the road | Calibration/time transform is wrong | Use same-clip calibration and `ego(t0)` to `ego(t)` alignment |
| `xstack fill not found` | FFmpeg is older | Remove `:fill=black` when all cells fill the canvas |

## 27. Actual Debugging Timeline

The verified reproduction passed through these gates:

1. Clone succeeded on the RTX 4060 Ti server.
2. Astral `uv` download stalled; PyPI mirror installation succeeded.
3. Python 3.12, PyTorch 2.8, CUDA, and project imports passed.
4. Direct Hugging Face timed out; a mirror retrieved `config.json`.
5. Official demo stopped on dataset authorization.
6. Correct account login fixed the dataset, then Cosmos authorization stopped the run.
7. Cosmos access was accepted and the 22 GB checkpoint downloaded.
8. Default inference stopped because FlashAttention was unavailable.
9. SDPA loaded all checkpoint shards.
10. The 8 GB GPU failed at model transfer with a genuine CUDA OOM.
11. Source instructions and selected caches moved to the A100.
12. A100 SDPA inference completed without stopping the existing GPU job.
13. Sliding videos, calibrated overlays, and the 2x2 composite were generated.

Solve errors in this order: network and account, gated access, dependencies, then GPU capacity.

## 28. Security Rules

- Never commit Hugging Face or GitHub tokens.
- Never publish SSH passwords or private keys.
- Use read-only Hugging Face tokens for downloads.
- Revoke any token accidentally exposed in chat or logs.
- Remove temporary transfer keys.
- Do not stop unrelated GPU jobs without permission.
- Do not redistribute gated weights or dataset archives.

## 29. Final Checklist

- [ ] At least 24 GB VRAM is available
- [ ] At least 80 GB disk space is free
- [ ] Python is 3.12
- [ ] `torch.cuda.is_available()` is `True`
- [ ] Alpamayo imports successfully
- [ ] Correct Hugging Face account is logged in
- [ ] Dataset, Alpamayo, and Cosmos access are accepted
- [ ] Main cache is approximately 21-22 GB
- [ ] SDPA is selected when FlashAttention is absent
- [ ] Completed windows are stored in `.pt` caches
- [ ] GPU model memory is released before rendering
- [ ] FFprobe reports correct duration and resolution
- [ ] Final MP4 is downloaded to the local computer

Once every item is checked, the complete clone-to-video workflow has been reproduced.
