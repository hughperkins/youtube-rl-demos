#!/bin/bash

set -e
set -x

IPADDRESS=$1
LOGFILEPATH=$2

./aws_pull.sh $IPADDRESS
python vizdoom/graph_log_batched.py \
     --in-logfile LOGFILEPATH && open vizdoom/graph.png
