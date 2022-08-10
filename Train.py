# External Library
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
import shutil
import time
import numpy as np
from matplotlib import pyplot as plt
import argparse
import subprocess
from subprocess import call, STDOUT

# Internal Library
from source.utilities import Get_Logger, Extract_from_LogFile

# Project Variables
from TORCH_OBJECTS import *

# Project Modules
import source.MODEL__Actor.grouped_actors2 as A_Module
import source.TRAIN_N_EVAL.Train_Grouped_Actors as T_Module
import source.TRAIN_N_EVAL.Evaluate__Grouped_Actors as E_Module


######################
# PARSE INPUTS
######################
parser = argparse.ArgumentParser(description='Train Model')

parser.add_argument('run_name', help='Name of Run. Used to generate output folder.')
parser.add_argument('problem_size', type=int, help='Integer size of problem data. Must match data found in data_dir')
parser.add_argument('train_data_dir', help='Dir where data is located. Must contain instances.npy and adjs.npy')
parser.add_argument('--m', type=int, help='Do mxm 2-step rollouts selecting top m options twice. -1 indicates no forced steps rollout. Defaults to -1', default=-1)
parser.add_argument('--deterministic', help='Changes the Train environments to deterministic.', action='store_true')
parser.add_argument('--test_data_dir', help='Dir where test set is located. Defaults to using train set for validation')
parser.add_argument('--epochs', type=int, help='Total Epochs to run. Defaults to 2000', default=2000)
parser.add_argument('--batch_size', type=int, help='Batch Size for training. Defaults to 64', default=64)
parser.add_argument('--test_batch_size', type=int, help='Batch Size for testing. Defaults to 256', default=256)
parser.add_argument('--embedding_dim', type=int, help='Embedding Dim size. Defaults to 128', default=128)
parser.add_argument('--key_dim', type=int, help='Key Dim size. Defaults to 16', default=16)
parser.add_argument('--head_num', type=int, help='Head Number. Defaults to 8', default=8)
parser.add_argument('--pretrained_dir', type=str, help="Filepath of pretrained model to start with. Must have matching hyperparams", default=None)
parser.add_argument('--encoder_layer_num', type=int, help='Defaults to 6', default=6)
parser.add_argument('--ff_hidden_dim', type=int, help='Defaults to 512', default=512)
parser.add_argument('--logit_clipping', type=int, help='Defaults to 10', default=10)
parser.add_argument('--actor_lr', type=float, help='Learning rate for actor. Defaults to 1e-4', default=1e-4)
parser.add_argument('--actor_wd', type=float, help='Actor weight decay. Defaults to 1e-6', default=1e-6)
parser.add_argument('--lr_decay_epoch', type=int, help='Defaults to 1', default=1)
parser.add_argument('--lr_decay_gamma', type=float, help='Defaults to 1.00', default=1.00)
parser.add_argument('--log_period_sec', type=int, help='Time between logging outputs in seconds. Defaults to 15.', default=15)
parser.add_argument('--dataset_mode', type=int, help="0 (Default): NPY Files, 1: From CSV, 2: Random", default=0)
parser.add_argument('--augment_data', help='Toggles on 8x data augmentation.', action='store_true')
parser.add_argument('--checkpoint_freq', type=int, help='Number of epochs between checkpoints. Defaults to 100', default=100)
parser.add_argument('--max_time_pen_mult', type=float, default=-1., help='Max time penalty = mult * problem_size. Defaults to -1')
parser.add_argument('--dist', default='normal', help='Use a right skewed dist instead of a uniform dist. Options are ["normal", "rayleigh"]')
args = parser.parse_args()

if args.test_data_dir is None:
    args.test_data_dir = args.train_data_dir


SAVE_FOLDER_NAME = "TRAIN_" + args.run_name.upper()
print(SAVE_FOLDER_NAME)

# Make Log File
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)

# Save used HYPER_PARAMS
hyper_param_save_path = f'{result_folder_path}/used_HYPER_PARAMS.txt'

train_progress_file = f'{result_folder_path}/train_output.csv'

with open(hyper_param_save_path, 'w') as f:
    if call(["git", "branch"], stderr=STDOUT, stdout=open(os.devnull, 'w')) == 0:
        # In a git repo, add commit id to hyper param logs
        git_commit_label = subprocess.check_output(
            ['git', 'describe', '--always']).strip()
        f.write(f'Git Commit: {str(git_commit_label)}\n\n')

    for k, v in args.__dict__.items():
        # Record hyper params
        f.write(f'{k.upper()} = {v}\n')

# Objects to Use
actor = A_Module.ACTOR(embedding_dim=args.embedding_dim,
                       encoder_layer_num=args.encoder_layer_num,
                       head_num=args.head_num,
                       key_dim=args.key_dim,
                       logit_clipping=args.logit_clipping,
                       ff_hidden_dim=args.ff_hidden_dim).to(device)
actor.optimizer = optim.Adam(
    actor.parameters(), lr=args.actor_lr, weight_decay=args.actor_wd)
actor.lr_stepper = lr_scheduler.StepLR(
    actor.optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay_gamma)

if args.pretrained_dir is not None and os.path.isdir(args.pretrained_dir):
    actor_path = os.path.join(args.pretrained_dir, "ACTOR_state_dic.pt")
    lrstep_path = os.path.join(args.pretrained_dir, "LRSTEP_state_dic.pt")
    optim_path = os.path.join(args.pretrained_dir, "OPTIM_state_dic.pt")
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.lr_stepper.load_state_dict(
        torch.load(lrstep_path, map_location=device))
    actor.optimizer.load_state_dict(
        torch.load(optim_path, map_location=device))


# Go
timer_start = time.time()
for epoch in range(1, args.epochs+1):

    log_package = {
        'epoch': epoch,
        'timer_start': timer_start,
        'logger': logger
    }

    # TRAIN
    T_Module.TRAIN(actor, args=args, **log_package)

    # EVAL
    E_Module.EVAL(actor, args=args,
                  train_progress_file=train_progress_file, **log_package)

    # Check Point
    if epoch % args.checkpoint_freq == 0:
        checkpoint_folder_path = '{}/CheckPoint_ep{:05d}'.format(
            result_folder_path, epoch)
        os.mkdir(checkpoint_folder_path)

        model_save_path = '{}/ACTOR_state_dic.pt'.format(
            checkpoint_folder_path)
        torch.save(actor.state_dict(), model_save_path)
        optimizer_save_path = '{}/OPTIM_state_dic.pt'.format(
            checkpoint_folder_path)
        torch.save(actor.optimizer.state_dict(), optimizer_save_path)
        lr_stepper_save_path = '{}/LRSTEP_state_dic.pt'.format(
            checkpoint_folder_path)
        torch.save(actor.lr_stepper.state_dict(), lr_stepper_save_path)


# Display results
exec_command_str = Extract_from_LogFile(result_folder_path, 'eval_result')
print(exec_command_str)
exec(exec_command_str)

plt.plot(0, 0)
plt.show()

plt.plot(eval_result)
plt.grid(True)

plt.savefig('{}/eval_result.jpg'.format(result_folder_path))

model_save_path = '{}/ACTOR_state_dic.pt'.format(result_folder_path)
torch.save(actor.state_dict(), model_save_path)
optimizer_save_path = '{}/OPTIM_state_dic.pt'.format(result_folder_path)
torch.save(actor.optimizer.state_dict(), optimizer_save_path)
lr_stepper_save_path = '{}/LRSTEP_state_dic.pt'.format(result_folder_path)
torch.save(actor.lr_stepper.state_dict(), lr_stepper_save_path)
