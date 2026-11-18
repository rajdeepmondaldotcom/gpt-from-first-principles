#!/usr/bin/env bash
# Copy this repository and the small files a lesson needs to the machine, run one command
# there, and bring results/ back.
#   aws/run.sh <host> python -m gpt.gpu
set -euo pipefail
host=${1:?the address of the machine, as aws/start.sh printed it}
shift
key=~/.ssh/gpt-from-first-principles
on() { ssh -i "$key" -o StrictHostKeyChecking=accept-new "ubuntu@$host" "$@"; }

git archive --format=tar HEAD | on 'mkdir -p course/data/cache && tar -x -C course'
# What the laptop's lessons made and the rented machine builds on: two tokenizers, the unseen
# text of each world, and the two trained models.
(cd data/cache && tar -c stories_vocab.json stories_val.bin story chess_vocab.json \
  chess_val.bin chess_val.txt chess) | on 'tar -x -C course/data/cache'
# The image's own Python has PyTorch built for its card. We add the two small libraries of
# the chess extra to it.
on "cd course && source /opt/pytorch/bin/activate && pip install -q chess zstandard && $*"
scp -q -i "$key" "ubuntu@$host:course/results/4*.json" results/
