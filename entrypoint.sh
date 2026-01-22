#!/usr/bin/env bash
set -e

echo "Start preprocessing data"
uv run inv preprocess-data

echo "Start training model"
uv run inv train