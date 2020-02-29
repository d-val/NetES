#!/usr/bin/env bash

CMD="python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --num_workers 120"

# until our command exits without an error, keep running it.
until $CMD; do
  echo "Command exited with code $?. Respawning..." >&2
  sleep 5
done
# run the same script again if it completed without an error code.
bash ./$0
