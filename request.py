# script to:
# request & deploy instances

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

import time

from os.path import dirname, realpath, join
import os

import sys

## usage
# first, install and  configure aws cli > https://docs.aws.amazon.com/cli/latest/userguide/tutorial-ec2-ubuntu.html
# check that you configured things correctly using: aws configure list
# python request.py 8

profile = os.environ['AWS_PROFILE']
keyname = os.environ['KEY_NAME']
group = os.environ['ES_GROUP']
spotprice = '5'

ami_list = {
    #"us-east-1": "YOUR_OWN",
}

instancetype = 'm4.16xlarge'

dayandtime = str(time.strftime("%Y-%m-%d")+'_'+time.strftime("%H:%M:%S"))

num_instances_total = int(sys.argv[1])

assert len(ami_list.keys()) == 1

region = list(ami_list.keys())[0]
print('region', region)

counter = 0
input('\nAbout to request '+str(num_instances_total)+" instances of type "+instancetype+" for region "+region+" !?! Continue?")

for instance_request in range(num_instances_total):
    s = boto3.Session(profile_name = profile) # Change to another user if necessary
    client = boto3.client('ec2', region_name=region)

    print('requesting 1 instance of type ', instancetype, ' for ', region)
    reservation = client.request_spot_fleet(
        SpotFleetRequestConfig={
            'SpotPrice': spotprice,
            'TargetCapacity': 1,
            'AllocationStrategy': 'lowestPrice',
            'IamFleetRole': os.environ['AWS_IAM_FLEET_ROLE'],
            'LaunchSpecifications': [
                {
                    'ImageId': ami_list[region],
                    'KeyName': keyname,
                    'InstanceType': instancetype
                }
            ],
        }
    )


    # with open('aws_requests_fast/{}_{}_{}_{}_spot_fleet_request.json'.format(region, group, counter, dayandtime), 'w') as f:
    with open('aws_requests/{}_{}_{}_{}_spot_fleet_request.json'.format(region, group, counter, dayandtime), 'w') as f:
        json.dump(reservation, f)
        print('\nSuccessfully requested spot instances. \n Request ID: {}'.format(reservation))

    counter = counter + 1
