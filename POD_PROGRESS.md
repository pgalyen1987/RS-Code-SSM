# Pod training snapshot (2026-05-09T14:12:14Z)

All training/benchmark Python jobs were **stopped** before this commit.

## Checkpoint files (large — not in Git)

| Path | Purpose |
|------|---------|
| `checkpoints/sft_clean/sft_latest.pt` | SFT (clean) latest |
| `checkpoints/grpo_clean/grpo_best.pt` | GRPO best so far |

Disk listing:

```
checkpoints/grpo_clean/:
total 12G
drwxrwxrwx 2 root root 2.9M May  9 14:05 .
drwxrwxrwx 6 root root 2.9M May  8 19:55 ..
-rw-rw-rw- 1 root root  12G May  9 14:06 grpo_best.pt

checkpoints/sft_clean/:
total 12G
drwxrwxrwx 2 root root 2.9M May  8 23:30 .
drwxrwxrwx 6 root root 2.9M May  8 19:55 ..
-rw-rw-rw- 1 root root  12G May  9 13:03 sft_latest.pt
```

## Hugging Face upload (run locally or on any machine with `HF_TOKEN`)

Model repo used elsewhere in this project: `pgalyen1987/RS-Code-SSM-1.6B`.

```bash
export HF_TOKEN=***   # read-only or write token from https://huggingface.co/settings/tokens

python3 << 'PY'
from pathlib import Path
from ssm.hf_checkpoint_sync import upload_checkpoint

repo = "pgalyen1987/RS-Code-SSM-1.6B"
upload_checkpoint(Path("checkpoints/sft_clean/sft_latest.pt"), repo, "training/sft_latest.pt")
upload_checkpoint(Path("checkpoints/grpo_clean/grpo_best.pt"), repo, "training/grpo_best.pt")
PY
```

Copy checkpoints off the pod first (e.g. `scp` / RunPod volume) if you are deleting the instance.

## Git

This commit captures: fixed `scripts/run_benchmarks.py`, `scripts/run_pipeline.sh`, verify log under `runs/`, and this manifest.
