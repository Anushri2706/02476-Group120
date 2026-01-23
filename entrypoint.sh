#!/usr/bin/env bash
set -e

export WANDB_SILENT=true
export WANDB_DISABLE_SERVICE=true

echo "Start preprocessing data"
uv run inv preprocess-data

echo "Start training model"
uv run inv train
