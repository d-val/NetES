#!/usr/bin/env bash

REDIS_CMD="redis-server ./redis_config/redis_master.conf"

until $REDIS_CMD; do
  echo "Redis worker ended with exit code $?. Respawning..." >&2
  sleep 1
done
