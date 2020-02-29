#!/usr/bin/env python
import sys
import time
import subprocess
import glob
import os
import atexit

TIME_TO_WAIT = 10 * 60

exp_file = sys.argv[1]
cmd = 'python -m es_distributed.main master \
      --master_socket_path /tmp/es_redis_master.sock \
      --log_dir ./logs --exp_file {}'.format(exp_file)
exp_name = os.path.splitext(os.path.basename(exp_file))[0]
exp_group_name = os.path.dirname(exp_file).split('/')[-1]
results_path = './logs/{}'.format(exp_group_name)


print("[watchdog] Running experiment {}....".format(exp_name))
time.sleep(2)


def time_since_last_result():
    """returns the number of seconds since the last result was posted.
    """
    list_of_files = glob.glob(os.path.join(results_path, '*_{}_*'.format(exp_group_name)))
    if len(list_of_files) == 0:
        print("Found no files in results directory: {}".format(results_path))
        return None
    print("checking directory:", results_path, "for exp name: {}".format(exp_group_name))
    print("checking files:", list_of_files)
    latest_file = max(list_of_files, key=os.path.getctime)
    num_lines = sum(1 for line in open(latest_file))
    last_time = os.path.getctime(latest_file)
    print("\t latest file for exp {} has {} lines.".format(exp_name, num_lines))
    print("\t Experiment {} last had a posted result for iteration {} at {}".format(exp_name, num_lines, last_time))

    return time.time() - last_time

def start_master():
    return subprocess.Popen(cmd, shell=True)

time_started = time.time()
os.setpgrp()

p = start_master()

while True:
    is_up = p.poll()
    if time_started < 10:
        continue

    if is_up is not None:
        print("[watchdog] Master process died, restarting....")
        p = start_master()

    t = time_since_last_result()

    if t is not None and t > TIME_TO_WAIT:
        print("[watchdog] Master hasn't posted results for a while, restarting...")
        print("[watchdog] Should start within 30 seconds...")
        p.kill()

    time.sleep(1)
