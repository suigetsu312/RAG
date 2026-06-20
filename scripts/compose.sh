#!/usr/bin/env bash
set -euo pipefail

COMPOSE_CMD=(
docker compose
--env-file ./docker/vllm.env
)

case "${1:-}" in
build)
"${COMPOSE_CMD[@]}" up --build
;;

up)
"${COMPOSE_CMD[@]}" up
;;

down)
"${COMPOSE_CMD[@]}" down
;;

restart)
"${COMPOSE_CMD[@]}" down
"${COMPOSE_CMD[@]}" up
;;

logs)
"${COMPOSE_CMD[@]}" logs -f
;;

status|ps)
"${COMPOSE_CMD[@]}" ps
;;

*)
echo "Usage: $0 {build|up|down|restart|logs|status}"
exit 1
;;
esac
