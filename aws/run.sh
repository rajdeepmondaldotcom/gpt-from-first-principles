#!/usr/bin/env bash
# Copy this repository's last commit and the files a lesson builds on to the machine, run one
# command there, and bring back what it made.
#   aws/run.sh <host> python -m gpt.gpu
# A command with && or quotes in it goes in as one quoted string.
# The command runs apart from this connection, so a laptop that sleeps does not stop it. With
# no command, the script follows the job that is running there and brings back its files:
#   aws/run.sh <host>
# VENV, if set, names a Python environment on the machine to run in (lesson 42 makes one).
set -euo pipefail
host=${1:?the address of the machine, as aws/start.sh printed it}
shift
key=~/.ssh/gpt-from-first-principles
door=(-i "$key" -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$key.hosts")
on() { ssh "${door[@]}" "ubuntu@$host" "$@"; }
there=ubuntu@$host:course

if [ $# -gt 0 ]; then
  # One job at a time: a second would share the card with the first and wipe its log.
  on '! pgrep -f "^bash job.sh"' > /dev/null ||
    { echo "a job is still running there. aws/run.sh $host follows it" >&2; exit 1; }
  # The commit goes over as one file, so each result names the commit it ran.
  bundle=$(mktemp)
  trap 'rm -f "$bundle"' EXIT
  git bundle create "$bundle" HEAD > /dev/null 2>&1
  scp -q "${door[@]}" "$bundle" "ubuntu@$host:course.bundle"
  on 'mkdir -p course/data/cache && cd course && git init -q &&
      git fetch -q ../course.bundle HEAD && git checkout -q -f --detach FETCH_HEAD'
  # What earlier lessons made and this machine does not have: the tokenizers, the text held
  # out of training, the trained models and their results. A file the machine already has
  # is left alone: it may be a result that has not come back yet.
  (cd data/cache && rsync -a --ignore-existing -e "ssh ${door[*]}" stories_vocab.json stories_val.bin \
    stories_train.bin story chess_vocab.json chess_val.bin chess_val.txt chess \
    $(ls -d story_big chess_big 2> /dev/null) "$there/data/cache/")
  rsync -a --ignore-existing -e "ssh ${door[*]}" results/4*.json "$there/results/" 2> /dev/null || true
  # The image's own Python has PyTorch built for its card. We add the chess extra's two libraries.
  # PYTHONUNBUFFERED has Python write each line to the log as it is printed.
  on 'cat > job.sh' << EOF
export PYTHONUNBUFFERED=1
cd course && source ${VENV:-/opt/pytorch}/bin/activate && pip install -q chess zstandard && $*
echo \$? > ~/job.status
EOF
  on 'rm -f job.status; nohup bash job.sh > job.log 2>&1 < /dev/null & sleep 1'
fi

# Follow the job to its end. If this connection drops, the job carries on.
on 'job=$(pgrep -o -f "^bash job.sh") && tail -n +1 -f --pid="$job" job.log || cat job.log'
rsync -a -e "ssh ${door[*]}" "$there/results/4*.json" results/ 2> /dev/null || true
on 'cd course/data/cache && tar -c $(ls -d story_big chess_big 2> /dev/null) || true' |
  tar -x -C data/cache
status=$(on 'cat job.status' 2> /dev/null) ||
  { echo "the job has not ended. aws/run.sh $host follows it" >&2; exit 1; }
exit "$status"
