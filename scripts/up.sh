#!/usr/bin/env bash
set -euo pipefail

docker compose \
--env-file ./docker/vllm.env \
up -d --build