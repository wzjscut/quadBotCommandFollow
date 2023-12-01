from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
from raisimGymTorch.env.bin.rsg_anymal import NormalSampler
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
try:
    import wandb
except:
    wandb = None

# task specification
task_name = "forward"

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='train')
parser.add_argument('-w', '--weight', type=str, default='')
parser.add_argument('-n', '--num', type=str, default=0)
args = parser.parse_args()
mode = args.mode
weight_path = args.weight 
num = args.num
start_i = 0

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs


actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])
#tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update
if wandb:
    resume_flag = False
    if mode == 'retrain':
        resume_flag = True
    wandb.init(project='command_loco', config=dict(cfg), name=task_name, save_code=True, resume=resume_flag)
    wandb.save(home_path + '/raisimGymTorch/env/envs/cmd_follow/Environment.hpp')

ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              )

if mode == 'retrain':
    start_i = load_param(weight_path + "/full_" + num + ".pt", env, actor, critic, ppo.optimizer, saver.data_dir)
    loaded_graph_flat = torch.jit.load(home_path + weight_path + "/policy_" + num + ".pt", map_location=torch.device(device))
    env.set_itr_number(int(num))
else:
    env.set_itr_number(0)

if wandb:
    wandb.watch(actor.architecture.architecture, log_freq=100)
    wandb.watch(critic.architecture.architecture, log_freq=100)
   
for update in range(start_i, 2000000):
    start = time.time()
    env.reset()
    reward_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")
        actor.save_deterministic_graph(saver.data_dir+"/policy_"+str(update)+'.pt', torch.rand(1, ob_dim).cpu())
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        parameters = np.zeros([0], dtype=np.float32)
        for param in actor.deterministic_parameters():
            parameters = np.concatenate([parameters, param.cpu().detach().numpy().flatten()], axis=0)
        np.savetxt(saver.data_dir+"/policy_"+str(update)+'.txt', parameters)
        loaded_graph = torch.jit.load(saver.data_dir+"/policy_"+str(update)+'.pt', map_location=torch.device(device))

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    # actual training
    reward_info_sum = 0
    for step in range(n_steps):
        obs = env.observe()
        action = ppo.act(obs)
        reward, dones = env.step(action)
        unscaled_reward_info = env.get_reward_info()
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_sum = reward_sum + np.sum(reward)
        reward_info_sum += np.sum(unscaled_reward_info, axis=0)

    actor.update()
    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()


    # for log 
    dx, dz, dyz, dpr, force, speed_reward, total_reward,_,_,_,_,_,_,_,_,_ = reward_info_sum / total_steps
    force = force * cfg['environment']['torque_coeff']

    obs = env.observe()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_sum / total_steps
    average_dones = done_sum / total_steps
    end = time.time()

    if (update % 50 == 0):
        if wandb:
            wandb.log({
            'dones': average_dones,
            'dx': dx,
            'dz': dz,
            'dyz': dyz,
            'dpr': dpr,
            'force': force,
            'speed_reward': speed_reward,
            'total_reward': total_reward
            })

    if (update % 200 == 0):
        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
        print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                        * cfg['environment']['control_dt'])))
        print('----------------------------------------------------\n')
