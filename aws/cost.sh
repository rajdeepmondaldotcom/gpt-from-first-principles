#!/usr/bin/env bash
# What this project's running machines have cost so far, by the price list: hours since launch
# times the hourly price, with each machine's disk and public address counted.
# Usage: AWS_PROFILE=<yours> AWS_REGION=us-east-1 aws/cost.sh [instance-id ...]
# With no ids it lists each machine tagged for the project that is starting or running.
set -euo pipefail
name=${PROJECT:-gpt-from-first-principles}
if [ $# -gt 0 ]; then
  which=(--instance-ids "$@")
else
  which=(--filters "Name=tag:project,Values=$name" "Name=instance-state-name,Values=pending,running")
fi
aws ec2 describe-instances "${which[@]}" --output text \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' |
  python3 -c '
import sys
from datetime import datetime, timezone

# Dollars an hour on demand in us-east-1, November 2026. A kind that is not here counts as 0.
price = {"g6e.xlarge": 1.861, "g6e.12xlarge": 10.493, "g5.xlarge": 1.006, "g5.12xlarge": 5.672}
price |= {"g6.xlarge": 0.805, "g6.12xlarge": 4.602}
extra = 200 * 0.08 / 730 + 0.005  # the 200 GB disk and the public address, for each hour
total = 0.0
for line in sys.stdin:
    machine, kind, state, launched = line.split()
    if state not in ("pending", "running"):
        print(f"{machine} {kind}: {state}, no longer charged")
        continue
    hours = (datetime.now(timezone.utc) - datetime.fromisoformat(launched)).total_seconds() / 3600
    dollars = hours * (price.get(kind, 0) + extra)
    total += dollars
    print(f"{machine} {kind} {state}: {hours:.2f} hours, ${dollars:.2f}, and ${price.get(kind, 0) + extra:.2f} more each hour")
print(f"in all, for machines still running: ${total:.2f} by the price list, before tax")
'
