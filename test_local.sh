#!/bin/sh

# runs a test experiment on your local machine.
# KILLS ANY REDIS SERVERS THAT ARE RUNNING. USE AT YOUR OWN RISK.
# will launch tmux sessions with a new redis instance and a new experiment.

# pass it the  path to an experiment configuration file.

### get latest code from github
#eval "$(ssh-agent -s)"
#ssh-add aws_github
#rm -rf collective-reinforcement-learning
#cd /home/ubuntu/collective-reinforcement-learning

killall redis-server
conda env create -f install/es-network.yml

NAME=es-network_`date "+%m_%d_%H_%M_%S"`
tmux new -s $NAME -d bash

tmux new-window -t $NAME -n redis
tmux send-keys -t $NAME:redis 'source activate es-network' C-m
tmux send-keys -t $NAME:redis 'bash scripts/watch_master_redis.sh' C-m
tmux split-window -t $NAME:redis
tmux send-keys -t $NAME:redis 'source activate es-network' C-m
tmux send-keys -t $NAME:redis right 'bash scripts/watch_worker_redis.sh' C-m
sleep 2

EXP_FILE=$1

# # we run bash to make sure you're in a bash shell
tmux new-window -t $NAME -n experiment -d bash
sleep 1

# # activate the conda env
tmux send-keys -t $NAME:experiment 'source activate es-network' C-m
tmux send-keys -t $NAME:experiment 'python scripts/watch_master_process.py ' "$EXP_FILE" C-m
#tmux send-keys -t $NAME:experiment 'python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --log_dir ./logs --exp_file '"$EXP_FILE" C-m
tmux split-window -t $NAME:experiment bash # same here
sleep 1

tmux send-keys -t $NAME:experiment right 'source activate es-network' C-m
tmux send-keys -t $NAME:experiment right 'bash scripts/watch_worker_process.sh' C-m
tmux a -t $NAME:experiment
