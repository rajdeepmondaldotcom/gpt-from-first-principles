#!/usr/bin/env bash
# Destroy each machine this course started in this region, and show that nothing of ours is left.
#   AWS_PROFILE=<profile> AWS_REGION=us-east-1 aws/stop.sh
set -euo pipefail
: "${AWS_PROFILE:?name the AWS profile to use}" "${AWS_REGION:?name the region}"
name=gpt-from-first-principles
ours="Name=tag:project,Values=$name"
live="Name=instance-state-name,Values=pending,running,stopping,stopped"

ids=$(aws ec2 describe-instances --filters "$ours" "$live" \
  --query 'Reservations[].Instances[].InstanceId' --output text)
if [ -n "$ids" ]; then
  aws ec2 terminate-instances --instance-ids $ids > /dev/null
  aws ec2 wait instance-terminated --instance-ids $ids
fi
aws ec2 delete-security-group --group-name "$name" 2> /dev/null || true
aws ec2 delete-key-pair --key-name "$name" > /dev/null
rm -f ~/.ssh/$name.hosts # the addresses of machines that no longer exist

echo "machines of ours still alive in $AWS_REGION (each line is one):"
aws ec2 describe-instances --filters "$ours" "$live" \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]' --output text
echo "disks of ours attached to nothing, which still cost money (each line is one):"
aws ec2 describe-volumes --filters "$ours" Name=status,Values=available \
  --query 'Volumes[].[VolumeId,Size]' --output text
echo "the door for SSH, if AWS has not let go of it yet (it costs nothing: run this again):"
aws ec2 describe-security-groups --filters "$ours" \
  --query 'SecurityGroups[].GroupId' --output text
