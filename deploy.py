#!/usr/bin/env python

# python deploy.py configurations/nips_rebuttal_configs --aws_request_dir aws_requests/nips_rebuttal_request

import datetime
import json
import os
import glob
import logging

import os, sys
import boto3
import json
import time
#make sure you have installed frabric3
from fabric.api import env
from fabric.tasks import execute
from fabric.operations import run, put
from fabric.api import hosts, env
from fabric.context_managers import cd, settings
import pandas as pd

import traceback

import time

from os.path import dirname, realpath, join
import os

import sys

import click


def highlight(x, fg='green'):
    if not isinstance(x, str):
        x = json.dumps(x, sort_keys=True, indent=2)
    click.secho(x, fg=fg)


def upload_archive(exp_name, archive_excludes, s3_bucket, skip_archive):
    import hashlib, os.path as osp, subprocess, tempfile, uuid, sys

    # Archive this package
    thisfile_dir = osp.dirname(osp.abspath(__file__))
    pkg_dir = osp.abspath(osp.join(thisfile_dir, '.'))
    highlight("Running tar from: {}".format(pkg_dir))
#    assert osp.abspath(__file__) == osp.join(pkg_parent_dir, pkg_subdir, 'deploy2.py'), 'You moved me!'

    # Run tar
#    tmpdir = tempfile.TemporaryDirectory()
    tmpdir = "/tmp"
    if skip_archive:
        highlight("Skipping archiving, using latest code...")
        # use latest archive
        archives = glob.glob(os.path.join(tmpdir, 'crl_*'))
        if len(archives) == 0:
            highlight("No existing archives found in {}...".format(tmpdir))
            input("\nPress return to continue and create a new code archive.")
            upload_archive(exp_name, archive_excludes, s3_bucket, False)
            return
        latest_file = max(archives, key=os.path.getctime)
        local_archive_path = latest_file
    else:
        local_archive_path = osp.join(tmpdir, 'crl_{}.tar.gz'.format(uuid.uuid4()))
        tar_cmd = ["tar", "-vzcf", local_archive_path]
        for pattern in archive_excludes:
            tar_cmd += ["--exclude", '{}'.format(pattern)]
        tar_cmd += ['../collective-reinforcement-learning']
        highlight("TAR CMD: {}".format(" ".join(tar_cmd)))

        if sys.platform == 'darwin':
            # Prevent Mac tar from adding ._* files
            env = os.environ.copy()
            env['COPYFILE_DISABLE'] = '1'
            subprocess.check_call(tar_cmd, env=env)
        else:
            subprocess.check_call(tar_cmd)

    # Construct remote path to place the archive on S3
    with open(local_archive_path, 'rb') as f:
        archive_hash = hashlib.sha224(f.read()).hexdigest()
    remote_archive_path = '{}/{}_{}.tar.gz'.format(s3_bucket, exp_name, archive_hash)

    # Upload
    upload_cmd = ["aws", "s3", "cp", local_archive_path, remote_archive_path]
    highlight(" ".join(upload_cmd))
    subprocess.check_call(upload_cmd)

    # presign_cmd = ["aws", "s3", "presign", remote_archive_path, "--expires-in", str(60 * 60 * 24 * 30)]
    presign_cmd = ["aws", "s3", "presign", remote_archive_path, "--expires-in", str(60 * 60 * 24)]
    highlight(" ".join(presign_cmd))
    remote_url = subprocess.check_output(presign_cmd).decode("utf-8").strip()
    return remote_url

def make_disable_hyperthreading_script():
    return """
# disable hyperthreading
# https://forums.aws.amazon.com/message.jspa?messageID=189757
for cpunum in $(
    cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list |
    sed 's/-/,/g' | cut -s -d, -f2- | tr ',' '\n' | sort -un); do
        echo 0 > /sys/devices/system/cpu/cpu$cpunum/online
done
"""

def make_download_script(code_url):
    return """
set -x
cd ~
rm -rf collective-reinforcement-learning
kill -9 $(pgrep redis)
tmux kill-server
wget -S '{code_url}' -O code.tar.gz
tar xvaf code.tar.gz
rm code.tar.gz
cd collective-reinforcement-learning
""".format(code_url=code_url)

def make_run_script(code_url, exp_str):
    CMD = """
    cat > ~/experiment.json <<< '{exp_str}'
    dtach -n `mktemp -u /tmp/dtach.XXXX` bash start_experiment.sh
    """.format(exp_str=exp_str)

    return """
    set -x
    {}
    {}
    """.format(make_download_script(code_url), CMD)

def run_on_nodes(host, cmd):
    with settings(host_string=host):
        res = run(cmd)


@click.command()
@click.argument('config_dir', type=click.Path(), required=True)
@click.option('--aws_request_dir', type=click.Path(), default='./aws_requests')
@click.option('--outfile', type=click.Path(), required=False)
@click.option('--archive_name', default="expt")
@click.option('--group', default='es_test_run')
@click.option('--archive_excludes', default=["aws_requests", "noncommittal", "configurations", "notes", "plots", "install", "videos", "./logs", "*blob", ".git", "keys", "results", "networkx_graph_files", "writing", "logs", "final_plots", "NK", "misc", "patent", "networkx_graph_files", "formalization", "patent", "overleaf", "rebuttal"])
# @click.option('--archive_excludes', default=["aws_requests", "noncommittal", "configurations", "notes", "plots", "install", "video", "./logs", "*blob", ".git", "keys", "results", "networkx_graph_files", "writing", "logs", "final_plots"])
@click.option('--skip_archive/--new-archive', default=False)
def deploy(config_dir, aws_request_dir, outfile, archive_name, group, archive_excludes, skip_archive):
    dayandtime = str(time.strftime("%Y-%m-%d")+'_'+time.strftime("%H:%M:%S"))
    exp_files = [expt for expt in os.listdir(config_dir) if expt.endswith('.json')]
    num_experiments = len(exp_files)
    highlight("Deploying:")
    highlight("Found {} experiments:".format(num_experiments))
    highlight("{}".format(exp_files))

    if not outfile:
        outfile = os.path.join(aws_request_dir, "nodeconfig_{}.json".format(dayandtime))
    highlight("nodeconfig.json file is: {}".format(outfile))

    highlight("Reading from request files to get DNS...")

    spot_dns = []
    current_dir = os.getcwd()
    # requests = os.listdir(os.path.join(current_dir, 'aws_requests/'))
    requests = os.listdir(os.path.join(current_dir, aws_request_dir))

    SpotFleetRequestId_instances_mapping = pd.DataFrame()
    instance_to_SpotFleetRequestId_dict = dict()
    counter = 0
    for request in requests:
        if not request.endswith('request.json'):
            continue
        left = len(requests) - counter
        highlight("Using file {} for looking up instances. {}".format(request, left))

        counter = counter + 1
        with open(os.path.join(aws_request_dir, request), 'r') as f:
            request_id = json.load(f)['SpotFleetRequestId']

        highlight("\nNow finding DNS for request {}".format(request))

        region = request.split("_")[0]
        session = boto3.Session(region_name=region)
        ec2 = session.resource('ec2')
        client = boto3.client('ec2', region_name=region)
        instances = client.describe_spot_fleet_instances(SpotFleetRequestId=request_id)
        instance_ids = [obj['InstanceId'] for obj in instances['ActiveInstances']]


        if (group == None):
            filters=[{'Name': 'instance-state-name', 'Values': ['running']},]
        else:
            filters=[{'Name': 'instance-state-name', 'Values': ['running']},
                     {'Name': 'instance-id', 'Values': instance_ids}]
        instances = ec2.instances.filter(Filters=filters)
        spot_dns = spot_dns + [instance.public_dns_name for instance in instances]
        # print("DNS names: ", spot_dns)
        print('[instances]', [instance.public_dns_name for instance in instances])

        for instance in instances:
            instance_to_SpotFleetRequestId_dict[instance.public_dns_name] = request_id

        SpotFleetRequestId_instances_mapping = SpotFleetRequestId_instances_mapping.append(
                                                {
                                                    'SpotFleetRequestId': request_id,
                                                    'instances': [instance.public_dns_name for instance in instances]
                                                }, ignore_index=True)

        highlight("saving node IPs to {}".format(outfile))
        with open(outfile, 'w') as f:
            json.dump({'nodes_ips': spot_dns}, f)

    SpotFleetRequestId_instances_mapping.to_csv(os.path.join(aws_request_dir,"SpotFleetRequestId_instances_mapping.csv"))
    # print(instance_to_SpotFleetRequestId_dict)

    print(outfile)
    with open(outfile, 'r') as f:
        spot_dns = json.load(f)['nodes_ips']
    highlight("Spot DNS: {}".format(spot_dns))
    hosts = spot_dns

    highlight("# Of Nodes available: {}\n\n".format(len(hosts)))


    # exp_files
    keyfile = os.environ['AWS_KEY_FILE']
    env.user = "ubuntu"
    ssh_file_path = join(dirname(realpath(__file__)), keyfile)
    env.key_filename = ssh_file_path
    counter = 0

    if len(hosts) < len(exp_files):
        highlight("SOME HOSTS DID NOT SPIN UP!! Request more hosts for the rest of the experiments")
        highlight("\n\nRunning the other expts.. ")


    input("\n\nLAST CHANCE BEFORE JOBS ARE DEPLOYED!!! CONTINUE?")

    highlight("...Making code tarball...")
    code_url = upload_archive(archive_name, archive_excludes, os.environ['S3_BUCKET'], skip_archive)
    highlight("Uploaded code to: {}".format(code_url))

    host_configuration_mapping = pd.DataFrame()
    for i in range(len(hosts)):
        # actual running code
        exp_file_path = os.path.join(config_dir, exp_files[i])
        highlight("...reading experiment file from {}".format(exp_file_path))
        with open(exp_file_path, 'r') as f:
            exp_str = json.load(f)

        try:
            execute(run_on_nodes, hosts[i], make_run_script(code_url, json.dumps(exp_str)))

            host_configuration_mapping = host_configuration_mapping.append(
                {
                    'host': hosts[i],
                    'configuration': exp_files[i],
                    # 'SpotFleetRequestId': instance_to_SpotFleetRequestId_dict[hosts[i]]
                }, ignore_index=True)

            ## move files just sent to in_progress folder:
            # if not os.path.exists(os.path.join(config_dir, 'in_progress')):
            #     os.makedirs(os.path.join(config_dir, 'in_progress'))
            highlight('moving file ' + exp_files[i] + ' to in_progress/ ..')
            os.rename(os.path.join(config_dir, exp_files[i]), os.path.join(config_dir, 'in_progress', exp_files[i]))

            highlight(">>RAN<< " + hosts[i] + " with experiment " +  exp_files[i], fg = "yellow" )
        except Exception as e:
            print("could not run on ", hosts[i], str(e))
            # logging.error(traceback.format_exc())


    host_configuration_mapping.to_csv(os.path.join(aws_request_dir,'host_configuration_mapping.csv'))

if __name__ == "__main__":
    deploy()
