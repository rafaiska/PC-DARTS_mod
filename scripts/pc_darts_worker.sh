#!/bin/bash

QUEUE_FILE="${HOME}/pc_darts_queue.txt"
LOCK_DIR=/tmp/pc_darts_queue.lock
PC_DARTS_COMMAND=""

function fetch_next_cmd() {
  while ! mkdir $LOCK_DIR 2>/dev/null; do
    sleep 1000
  done

  read -r PC_DARTS_COMMAND < "$QUEUE_FILE"
  sed -i '1d' "$QUEUE_FILE"

  rm -r $LOCK_DIR
}

module load singularity/3.7.1
echo "###HARDWARE INFO:"
lshw
echo "###ENV INFO:"
env

while [ -s "$QUEUE_FILE" ]; do
  fetch_next_cmd
  echo "###RUNNING COMMAND: $PC_DARTS_COMMAND"
  echo "###START: $(date)"
  "singularity exec --nv -B ./workspace/mount:/workspace/mount ./pc_darts_p3_latest.sif $PC_DARTS_COMMAND"
  echo "###END: $(date)"
done