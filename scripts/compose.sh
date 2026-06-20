#!/usr/bin/env bash
set -euo pipefail

COMPOSE_CMD=(
docker compose
--env-file ./docker/vllm.env
)

case "${1:-}" in
up)
"${COMPOSE_CMD[@]}" up -d --build
;;

down)
"${COMPOSE_CMD[@]}" down
;;

restart)
"${COMPOSE_CMD[@]}" down
"${COMPOSE_CMD[@]}" up -d --build
;;

logs)
"${COMPOSE_CMD[@]}" logs -f
;;

status|ps)
"${COMPOSE_CMD[@]}" ps
;;

*)
echo "Usage: $0 {up|down|restart|logs|status}"
exit 1
;;
esac
