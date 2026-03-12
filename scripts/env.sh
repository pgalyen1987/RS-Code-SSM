#!/usr/bin/env bash
# scripts/env.sh — establish repo-relative dirs and Kaggle-friendly defaults
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}"
WORK_DIR="${WORK_DIR:-${KAGGLE_WORKING:-/kaggle/working}}"
# EpiChat: in-repo at $REPO_ROOT/epichat
EPICHAT_DIR="${EPICHAT_DIR:-$REPO_ROOT/epichat}"
# Model storage: ~/.ssm/models locally, $WORK_DIR/models on Kaggle
MODEL_DIR="${MODEL_DIR:-$([ -d /kaggle ] && echo "$WORK_DIR/models" || echo "$HOME/.ssm/models")}"
CONFIG_DIR="${CONFIG_DIR:-$([ -d /kaggle ] && echo "$WORK_DIR/.ssm" || echo "$HOME/.ssm")}"
TMP_DIR="${TMP_DIR:-/tmp}"
export REPO_ROOT DATA_DIR CHECKPOINT_DIR WORK_DIR EPICHAT_DIR MODEL_DIR CONFIG_DIR TMP_DIR