import errno
import json
import logging
import os
import sys
import glob
import h5py

import click

import os.path as osp
from .dist import RelayClient
from .es import run_master, run_master_from_snapshot, run_worker, SharedNoiseTable


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)


@cli.command()
@click.option('--exp_str')
@click.option('--exp_file')
@click.option('--master_socket_path', required=True)
@click.option('--log_dir')
@click.option('--snapshot/--no-snapshot', default=True)
@click.option('--snapshot_file')
@click.option('--theta_pickle')
def master(exp_str, exp_file, master_socket_path, log_dir, snapshot, snapshot_file, theta_pickle):
    import pickle

    if exp_str:
        exp = json.loads(exp_str)
    elif exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())

    exp_name = exp['config']['experiments_filename'].split('.')[0]
    exp_group_name = exp['config']['experiment_group_name']
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())

    snapshot_dir = "./snapshots/{}".format(exp_group_name)
    print("\nChecking {} for snapshots matching experiment name {} ...".format(snapshot_dir, exp_name))
    snapshots = glob.glob(os.path.join(snapshot_dir, '*_{}_*.h5'.format(exp_name)))

    if theta_pickle:
        print("Setting all agents to parameters found in pickle file {}".format(theta_pickle))
        assert (exp_str is None) != (exp_file is None), 'Must provide exp_str xor exp_file to the master'
        if exp_str:
            exp = json.loads(exp_str)
        elif exp_file:
            with open(exp_file, 'r') as f:
                exp = json.loads(f.read())
        else:
            assert False

        with open(theta_pickle, 'rb') as f:
            best_agent_thetas = pickle.load(f)

        # TODO: change to read from exp file
        num_agents = 1000

        thetas = [best_agent_thetas for agent in range(num_agents)]
        all_attrs = {'num_agents': num_agents,
                     'iteration': 0}

        run_master({'unix_socket_path': master_socket_path}, log_dir, exp,
                   exp_name, exp_group_name, snapshot_data=(thetas, all_attrs))

    elif snapshot and len(snapshots) > 0:
        # find latest snapshot file that matches our experiment
        latest_snapshot = max(snapshots, key=os.path.getctime)
        print("connecting to redis, socket path:", master_socket_path)
        if not snapshot_file:
            print("Running experiment from latest snapshot: ".format(latest_snapshot))
            snapshot_file_name = latest_snapshot
        else:
            print("Running experiment from snapshot file {}".format(snapshot_file))
            snapshot_file_name = snapshot_file

        run_master_from_snapshot({'unix_socket_path': master_socket_path},
            snapshot_file_name,
            log_dir)

    elif snapshot and snapshot_file:
        # find latest snapshot file that matches our experiment
        print("connecting to redis, socket path:", master_socket_path)
        print("Running experiment from snapshot file {}".format(snapshot_file))
        snapshot_file_name = snapshot_file

        run_master_from_snapshot({'unix_socket_path': master_socket_path},
            snapshot_file_name,
            log_dir)
    else:
        if snapshot and len(snapshots) == 0:
            print("-------\nNo snapshots found! Starting a NEW experiment.\n-------\n")
        # Start the master
        assert (exp_str is None) != (exp_file is None), 'Must provide exp_str xor exp_file to the master'
        if exp_str:
            exp = json.loads(exp_str)
        elif exp_file:
            with open(exp_file, 'r') as f:
                exp = json.loads(f.read())
        else:
            assert False
        print("connecting to redis, socket path:", master_socket_path)
        run_master({'unix_socket_path': master_socket_path}, log_dir, exp, exp_name, exp_group_name)

@cli.command()
@click.option('--snapshot')
@click.option('--master_socket_path', required=True)
@click.option('--log_dir')
def master_snapshot(snapshot, master_socket_path, log_dir):
    # Start the master
    assert (osp.exists(snapshot) and snapshot.endswith('.h5')), 'Must provide a h5py snapshot file'
    print("Continuing from a saved snapshot file at {}".format(snapshot))
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)
    exp_name = os.path.splitext(os.path.basename(exp_file))[0]
    exp_group_name = os.path.dirname(exp_file).split('/')[-1]
    print("connecting to redis, socket path:", master_socket_path)
    run_master_from_snapshot({'unix_socket_path': master_socket_path}, snapshot, log_dir)


@cli.command()
@click.option('--master_host', required=True)
@click.option('--master_port', default=6379, type=int)
@click.option('--relay_socket_path', required=True)
@click.option('--num_workers', type=int, default=0)
def workers(master_host, master_port, relay_socket_path, num_workers):
    # Start the relay
    master_redis_cfg = {'host': master_host, 'port': master_port}
    relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    if os.fork() == 0:
        logging.info("Creating relay client...")
        RelayClient(master_redis_cfg, relay_redis_cfg).run()
        return
    # Start the workers
    noise = SharedNoiseTable()  # Workers share the same noise
    num_workers = num_workers if num_workers else os.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    for _ in range(num_workers):
        if os.fork() == 0:
            run_worker(relay_redis_cfg, noise=noise)
            return
    os.wait()


if __name__ == '__main__':
    cli()
