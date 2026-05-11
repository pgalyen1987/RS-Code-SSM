#!/usr/bin/env python3
"""
Collect SFT/GRPO log files into runs/<run_id>/:
  - sft_metrics.csv     step,epoch,loss,aux,gnorm,lr
  - grpo_metrics.csv    step,loss,reward,kl,lr
  - sft_samples.txt     qualitative samples over training
  - grpo_samples.txt
  - manifest.json       paths, timestamps, final metrics
Upload everything to HF if HF_TOKEN + HF_MODEL_REPO are set.
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


# ── parsers ──────────────────────────────────────────────────────────────────

_SFT_STEP = re.compile(
    r"epoch=(\d+)\s+step=(\d+)/\d+\s+loss=(\S+)\s+aux=(\S+)\s+gnorm=(\S+).*lr=(\S+)"
)
_GRPO_STEP = re.compile(
    r"step=(\d+).*loss=(\S+).*reward=(\S+).*kl=(\S+).*lr=(\S+)"
)
_SAMPLE = re.compile(r"\[SAMPLE step=(\d+)\](.*)")


def parse_sft(log_path: Path):
    metrics, samples = [], []
    for line in log_path.read_text().splitlines():
        m = _SFT_STEP.search(line)
        if m:
            epoch, step, loss, aux, gnorm, lr = m.groups()
            metrics.append(dict(epoch=epoch, step=step, loss=loss,
                                aux=aux, gnorm=gnorm, lr=lr))
            continue
        m = _SAMPLE.search(line)
        if m:
            samples.append(f"[step={m.group(1)}]{m.group(2)}")
    return metrics, samples


def parse_grpo(log_path: Path):
    metrics, samples = [], []
    for line in log_path.read_text().splitlines():
        m = _GRPO_STEP.search(line)
        if m:
            step, loss, reward, kl, lr = m.groups()
            metrics.append(dict(step=step, loss=loss,
                                reward=reward, kl=kl, lr=lr))
            continue
        m = _SAMPLE.search(line)
        if m:
            samples.append(f"[step={m.group(1)}]{m.group(2)}")
    return metrics, samples


# ── HF upload ────────────────────────────────────────────────────────────────

def upload_to_hf(run_dir: Path, repo_id: str, token: str):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        for f in run_dir.iterdir():
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f"runs/{run_dir.name}/{f.name}",
                repo_id=repo_id,
                repo_type="model",
            )
        print(f"[HF] Uploaded {run_dir.name}/ → {repo_id}", flush=True)
    except Exception as e:
        print(f"[HF] Upload failed: {e}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--sft-log", default=None, help="explicit SFT log path")
    ap.add_argument("--grpo-log", default=None, help="explicit GRPO log path")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    out_root = Path(args.out_dir)
    run_id   = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir  = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Find logs
    def latest(pattern):
        files = sorted(logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None

    sft_log  = Path(args.sft_log)  if args.sft_log  else latest("sft_reasoning_*.log")
    grpo_log = Path(args.grpo_log) if args.grpo_log else latest("grpo_*.log")

    manifest = {"run_id": run_id, "created_utc": datetime.utcnow().isoformat()}

    # SFT
    if sft_log and sft_log.exists():
        metrics, samples = parse_sft(sft_log)
        if metrics:
            csv_path = run_dir / "sft_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
                w.writeheader(); w.writerows(metrics)
            manifest["sft_log"]     = str(sft_log)
            manifest["sft_steps"]   = len(metrics)
            manifest["sft_final_loss"] = metrics[-1]["loss"]
            print(f"[collect] SFT: {len(metrics)} steps → {csv_path}", flush=True)
        if samples:
            (run_dir / "sft_samples.txt").write_text("\n".join(samples))
            print(f"[collect] SFT: {len(samples)} samples saved", flush=True)
    else:
        print("[collect] No SFT log found", flush=True)

    # GRPO
    if grpo_log and grpo_log.exists():
        metrics, samples = parse_grpo(grpo_log)
        if metrics:
            csv_path = run_dir / "grpo_metrics.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
                w.writeheader(); w.writerows(metrics)
            manifest["grpo_log"]        = str(grpo_log)
            manifest["grpo_steps"]      = len(metrics)
            manifest["grpo_final_loss"] = metrics[-1]["loss"]
            manifest["grpo_final_reward"] = metrics[-1]["reward"]
            print(f"[collect] GRPO: {len(metrics)} steps → {csv_path}", flush=True)
        if samples:
            (run_dir / "grpo_samples.txt").write_text("\n".join(samples))
    else:
        print("[collect] No GRPO log found", flush=True)

    # Manifest
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[collect] Run saved → {run_dir}", flush=True)

    # HF upload
    token   = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_MODEL_REPO")
    if token and repo_id:
        upload_to_hf(run_dir, repo_id, token)
    else:
        print("[collect] HF_TOKEN/HF_MODEL_REPO not set — skipping upload", flush=True)


if __name__ == "__main__":
    main()
