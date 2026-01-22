#!/usr/bin/env bash

echo "Start preprocessing data"
uv run inv preprocess-data

echo "Start training model"
uv run inv train
