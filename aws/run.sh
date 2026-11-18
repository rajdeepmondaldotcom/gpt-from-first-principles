#!/usr/bin/env bash
# Copy this repository's last commit and the files a lesson builds on to the machine, run one
# command there, and bring back what it made.
#   aws/run.sh <host> python -m gpt.gpu
set -euo pipefail
host=${1:?the address of the machine, as aws/start.sh printed it}
shift
key=~/.ssh/gpt-from-first-principles
on() { ssh -i "$key" -o StrictHostKeyChecking=accept-new "ubuntu@$host" "$@"; }

# The commit goes over as one file and is cloned there, so each result names the commit it ran.
bundle=$(mktemp)
git bundle create "$bundle" HEAD > /dev/null 2>&1
scp -q -i "$key" -o StrictHostKeyChecking=accept-new "$bundle" "ubuntu@$host:course.bundle"
rm "$bundle"
on '[ -d course ] || git clone -q course.bundle course
    cd course && git pull -q ../course.bundle HEAD && mkdir -p data/cache'
# What earlier lessons made and this machine does not have: the tokenizers, the text held out
# of training, the trained models and their results.
(cd data/cache && tar -c stories_vocab.json stories_val.bin stories_train.bin story \
  chess_vocab.json chess_val.bin chess_val.txt chess $(ls -d story_big chess_big 2> /dev/null)) |
  on 'tar -x -C course/data/cache'
scp -q -i "$key" results/4*.json "ubuntu@$host:course/results/" 2> /dev/null || true
# The image's own Python has PyTorch built for its card. We add the chess extra's two libraries.
on "cd course && source /opt/pytorch/bin/activate && pip install -q chess zstandard && $*"
scp -q -i "$key" "ubuntu@$host:course/results/4*.json" results/
on 'cd course/data/cache && tar -c $(ls -d story_big chess_big 2> /dev/null)' | tar -x -C data/cache
