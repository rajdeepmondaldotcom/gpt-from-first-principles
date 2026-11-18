#!/usr/bin/env bash
# Start one GPU machine and print its id and address.
#   AWS_PROFILE=<profile> AWS_REGION=us-east-1 aws/start.sh g6e.xlarge 300
# The second number is a time limit in minutes. When it passes the machine shuts itself down,
# and a machine started this way is destroyed when it shuts down. So is its disk.
set -euo pipefail
: "${AWS_PROFILE:?name the AWS profile to use}" "${AWS_REGION:?name the region}"
kind=${1:?the machine, such as g6e.xlarge} minutes=${2:?a time limit in minutes}
name=gpt-from-first-principles key=~/.ssh/$name
tag="{Key=project,Value=$name}"

# The image: Ubuntu with the NVIDIA driver and PyTorch, which AWS publishes under this name.
image=$(aws ec2 describe-images --owners amazon --output text \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.14 (Ubuntu 26.04)*" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId')

# A key made on this computer. Only its public half goes to AWS.
[ -f "$key" ] || ssh-keygen -q -t ed25519 -N "" -f "$key"
aws ec2 describe-key-pairs --key-names "$name" > /dev/null 2>&1 ||
  aws ec2 import-key-pair --key-name "$name" --public-key-material "fileb://$key.pub" \
    --tag-specifications "ResourceType=key-pair,Tags=[$tag]" > /dev/null

# A door for SSH, open to this computer's address and no other.
group=$(aws ec2 describe-security-groups --group-names "$name" \
  --query 'SecurityGroups[0].GroupId' --output text 2> /dev/null || true)
if [ -z "$group" ] || [ "$group" = None ]; then
  group=$(aws ec2 create-security-group --group-name "$name" --description "SSH for $name" \
    --tag-specifications "ResourceType=security-group,Tags=[$tag]" --query GroupId --output text)
fi
mine=$(curl -sf https://checkip.amazonaws.com) # with no address the script stops here
aws ec2 authorize-security-group-ingress --group-id "$group" --protocol tcp --port 22 \
  --cidr "$mine/32" > /dev/null 2>&1 || true # open already

# AWS_ZONE, if set, names the zone to start in: a zone can run out of a kind of machine.
id=$(aws ec2 run-instances --image-id "$image" --instance-type "$kind" --key-name "$name" \
  ${AWS_ZONE:+--placement "AvailabilityZone=$AWS_ZONE"} \
  --security-group-ids "$group" --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data "$(printf '#!/bin/bash\nshutdown -h +%s\n' "$minutes")" \
  --tag-specifications "ResourceType=instance,Tags=[$tag,{Key=Name,Value=$name}]" \
  "ResourceType=volume,Tags=[$tag]" \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$id"
host=$(aws ec2 describe-instances --instance-ids "$id" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
echo "$id $host"
