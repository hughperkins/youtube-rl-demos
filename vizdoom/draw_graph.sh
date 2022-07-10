#!/bin/bash

ssh -i ~/ec2/2handmbp2022.pem ubuntu@52.23.208.70 cp /home/ubuntu/git/youtube-demos/log.txt /home/ubuntu/git/youtube-demos/pull/log.txt && (cd pull && rsync -av -e "ssh -i ~/ec2/2handmbp2022.pem" ubuntu@52.23.208.70:/home/ubuntu/git/youtube-demos/pull/ ./ ); python vizdoom/graph_log.py --in-logfile pull/log.txt && open vizdoom/graph.png
