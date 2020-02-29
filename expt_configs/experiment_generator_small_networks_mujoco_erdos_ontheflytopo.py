import itertools
import json
import os.path as osp
import os
import time

noise_stdev_list = [0.02]
strategy_list = ['Neighborhood']
network_type_list = ['smallworld']
comms_noise_stdev_list = [0.001]
return_proc_mode_list = ["centered_rank"]
num_threads = 120
snapshot_freq = 10000
same_param_start = [True, False]
on_the_fly_topology = [True, False]

# erdos
# b, p


num_agents_list = [100, 200, 300]

envs = ["Humanoid-v1",
        "Ant-v1",
        "HalfCheetah-v1",
        "Hopper-v1"]

erdos_exps = {"group_name": "mujoco_smallnets",
              "parameter_list": [(0.2, 0.8)], # (network param, broadcast)
              "environments": envs,
              "network_types": ["erdos"]
              }

generated_exps = {"group_name": "mujoco_smallnets",
                  "parameter_list": [287],
                  "environments": envs,
                  "network_types": ["generated387"]
              }

# full_exps = {"group_name": "mujoco_full_nobroadcast",
full_exps = {"group_name": "erdos_small_nets",
             "parameter_list": [""],
             "environments": envs,
             "network_types": ["fully-connected"]
}

scalefree_exps = {"group_name": "mujoco_smallnets",
                  "parameter_list": ["m4"],
                  "environments": envs,
                  "network_types": ["scalefree"]
}

erdos_params_envs, generated_params_envs, full_params_envs = [],[],[]

erdos_params_envs = [(a[0], a[1], b,c) for a,b,c in
                     list(itertools.product(erdos_exps['parameter_list'],
                                            erdos_exps['environments'],
                                            num_agents_list))]

generated_params_envs = list(itertools.product(generated_exps['parameter_list'],
                                               generated_exps['environments'],
                                               num_agents_list))

full_params_envs = list(itertools.product(full_exps['parameter_list'],
                                          full_exps['environments'],
                                          num_agents_list))

scalefree_params_envs = list(itertools.product(scalefree_exps['parameter_list'],
                                               scalefree_exps['environments'],
                                               num_agents_list))

print(scalefree_params_envs)



erdos_exps = []

generated_exps = []

full_exps = [(t[2], #num_agents
              noise_stdev_list[0],
              # 0.0, #broadcast
              0.9, #broadcast
              t[1], # env
              strategy_list[0],
              'erdos',
              0.2, #t[0], #['network_args']['p']
              comms_noise_stdev_list[0],
              return_proc_mode_list[0],
              full_exps['group_name'],
              # same_param_start[1], # same_param_start = [True, False]
              same_param_start[0], # same_param_start = [True, False]
              # on_the_fly_topology[0] #True
              on_the_fly_topology[1] #False on_the_fly_topology = [True, False]
              ) for t in full_params_envs]

scalefree_exps = []


with open('experiment_template.json', 'r') as f:
    template = json.load(f)

input('about to generate {} erdos and {} and generated and {} full-connected experiments! continue !?!'.format(len(erdos_exps), len(generated_exps), len(full_exps)))

for e in erdos_exps + generated_exps + full_exps + scalefree_exps:
    print(e)
    if not osp.exists(e[-1]):
        os.makedirs(e[-1])

# all_exp = erdos_exps + generated_exps + full_exps + scalefree_exps
all_exp = full_exps

for experiment_parameters in all_exp:
    json_file_name = "{0}_{1}_{2}{3}_numA{4}_broadcastp{5}_threads{8}_samepar{6}_topolran_{7}_{9}.json".format(
        experiment_parameters[3],
        experiment_parameters[4],
        experiment_parameters[5],
        experiment_parameters[6],
        experiment_parameters[0],
        experiment_parameters[2],
        experiment_parameters[10],
        experiment_parameters[11],
        num_threads,
        time.strftime("%H:%M:%S"))
    print(json_file_name)

    template['config']['experiments_filename']              = json_file_name
    template['policy']['args']['same_param_start']          = True #experiment_parameters[10]
    template['config']['snapshot_freq']                     = snapshot_freq
    template['config']['experiment_group_name']             = experiment_parameters[9]
    template['config']['num_agents']                        = experiment_parameters[0]
    template['config']['noise_stdev']                       = experiment_parameters[1]
    template['config']['broadcast_prob']                    = 0.8 #experiment_parameters[2]
    template['env_id']                                      = experiment_parameters[3]
    template['agent_update']['strategy']                    = experiment_parameters[4]
    template['agent_update']['args']['network_type']        = 'erdos' #experiment_parameters[5]
    template['agent_update']['args']['network_args']['p']   = 0.2 #experiment_parameters[6]
    template['agent_update']['comms_noise_stdev']           = experiment_parameters[7]
    template['agent_update']['return_proc_mode']            = experiment_parameters[8]
    template['agent_update']['on_the_fly_topology']         = True #experiment_parameters[11]
    template['config']['num_threads']                       = num_threads
    if 'erdos' in experiment_parameters[5]:
        template['agent_update']['args']['network_file'] = "erdos_n_{}_p_{}.pickle".format(experiment_parameters[0], str(experiment_parameters[6]))
    elif 'generated' in experiment_parameters[5]:
        template['agent_update']['args']['network_file'] = str(experiment_parameters[6])+"_engineered.pickle"
    elif 'scalefree' in experiment_parameters[5]:
        template['agent_update']['args']['network_file'] = "scale_free_{}_m4_seed10.pickle".format(str(experiment_parameters[0]))
    elif 'full' in experiment_parameters[5]:
        # doesn't matter
        template['agent_update']['args']['network_file']        = "fully_{}.pickle".format(experiment_parameters[0])




    # json_file_name = str(experiment_parameters[3]+"_"+experiment_parameters[4])+"_"+str(experiment_parameters[5])+"_p"+str(experiment_parameters[6])+"_num_agents"+str(experiment_parameters[0])+"_broadcastp_"+str(experiment_parameters[2])+"_threads_"+str(num_threads)+".json"
    print(json_file_name, 'generated')


    print("saving expt to {}/{}".format(experiment_parameters[-1], json_file_name))
    # with open(experiment_parameters[-1]+'/'+json_file_name, 'w') as f:
    # with open('full_random_param_on_the_fly_topology_broadcast/'+json_file_name, 'w') as f:
    with open('erdos_small_nets/'+json_file_name, 'w') as f:
        json.dump(template, f, indent=4)
    # break
