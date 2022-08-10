
"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import os

# For Logging
import time

# For debugging

from TORCH_OBJECTS import *

from source.utilities import Average_Meter, augment_data_by_8_fold
from source.td_opswtw import GROUP_ENVIRONMENT, DATA_LOADER__FROM_NPY, DATA_LOADER__FROM_FILE


########################################
# TRAIN
########################################

def TRAIN(grouped_actor, args, epoch, timer_start, logger):

    grouped_actor.train()

    dist_AM = Average_Meter()
    avg_len_AM = Average_Meter()
    actor_loss_AM = Average_Meter()
    group_unique_AM = Average_Meter()
    train_loader = None
    if args.dataset_mode == 0:
        train_loader = DATA_LOADER__FROM_NPY(data_dir=args.train_data_dir,
                                             batch_size=args.batch_size)
    elif args.dataset_mode == 1:
        train_loader = DATA_LOADER__FROM_FILE(data_dir=os.path.join(args.train_data_dir, 'instances/'),
                                              adj_dir=os.path.join(args.train_data_dir, 'adjs/'),
                                              batch_size=args.batch_size)
    else:
        raise NotImplemented("Other Modes not yet implemented")
        # TODO: Fix Random Loader
    # train_loader = DATA_LOADER__RANDOM(num_sample=TRAIN_DATASET_SIZE,
    #                                         num_nodes=args.problem_size,
    #                                         batch_size=args.batch_size)
    if args.m == -1:
        group_s = args.problem_size
    else:
        group_s = args.m * args.m

    n_instances = len(train_loader.dataset)
    if bool(args.augment_data):
        n_instances *= 8
    logger_start = time.time()
    episode = 0
    for xs, adjs in train_loader:
        # xs.shape = (batch, problem, 7)
        # adjs.shape = (batch, problem, problem)

        if bool(args.augment_data):
            xs = augment_data_by_8_fold(xs)
            adjs = adjs.repeat(8, 1, 1)

        batch_s = xs.size(0)

        episode = episode + batch_s

        # Actor Group Move
        ###############################################
        env = GROUP_ENVIRONMENT(xs, adjs, deterministic=args.deterministic, max_time_pen_mult=args.max_time_pen_mult,
                                dist=args.dist)
        if args.m == -1:
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            group_reward = Tensor(np.zeros((batch_s, group_s)))
        else:
            group_state, reward, done = env.reset(group_size=group_s)
            grouped_actor.reset(group_state)

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            group_reward = Tensor(np.zeros((batch_s, group_s)))
            # shape = (batch, group)

            ########
            # Step 1
            action_probs = grouped_actor.get_action_probabilities(group_state)
            # shape = (batch, group, problem)

            # Use only the first index (because all tours in group are in same state at depot)
            _, action_indices = action_probs[:, 0, :].topk(k=args.m, dim=1)
            action = action_indices.repeat_interleave(args.m, dim=1)
            # shape = (batch, m)
            group_state, reward, done = env.step(action + 1, noise_clump_size=args.m)
            group_reward += reward

            # Prob list
            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)

            ########
            # Step 2
            action_probs = grouped_actor.get_action_probabilities(group_state)
            diff_tour_indices = [i for i in range(0, group_s, args.m)]
            _, action_indices = action_probs[:, diff_tour_indices, :].topk(k=args.m, dim=2)
            # shape = (batch, m, m)
            action = action_indices.reshape(batch_s, group_s)
            # shape = (batch, group=mxm)
            action[group_state.finished] = 0

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)

            action[chosen_action_prob == 0] = 0
            group_state, reward, done = env.step(action + 1)
            group_reward += reward

            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)

            # Prob list
            chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            # shape = (batch, group, x)


        while not done:
            action_probs = grouped_actor.get_action_probabilities(group_state)
            # shape = (batch, group, problem+1)
            action = action_probs.reshape(batch_s * group_s, -1).multinomial(1)\
                .squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            action[group_state.finished] = 0  # stay at depot, if you are finished
            group_state, reward, done = env.step(action+1)
            group_reward += reward

            batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
            # shape = (batch, group)
            chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
            group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            # shape = (batch, group, x)

        # LEARNING - Actor
        ###############################################
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)

        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()

        grouped_actor.optimizer.zero_grad()
        loss.backward()
        grouped_actor.optimizer.step()

        # RECORDING
        ###############################################
        tour_cust_length = (group_state.selected_node_list != 1).sum(dim=2) + 1
        avg_len_AM.push(tour_cust_length)
        mean_reward = group_reward.mean(dim=1)
        max_reward, _ = group_reward.max(dim=1)
        dist_AM.push(mean_reward)
        actor_loss_AM.push(group_loss.detach())
        group_unique_AM.push(Tensor([group_reward[0].unique().numel()]), n_for_rank_0_tensor=1)

        # LOGGING
        ###############################################
        if (time.time()-logger_start > args.log_period_sec) or (episode == n_instances):
            timestr = time.strftime("%H:%M:%S", time.gmtime(time.time()-timer_start))
            log_str = 'Ep:{:03d}-{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  Avg.dist:{:5f}  Avg.tour_len:{:f}  Sampled group unique:{:5f}' \
                .format(epoch, episode, episode/n_instances * 100,
                        timestr, actor_loss_AM.result(), dist_AM.result(), avg_len_AM.result(), group_unique_AM.result()/group_s)
            logger.info(log_str)
            logger_start = time.time()

    # LR STEP, after each epoch
    grouped_actor.lr_stepper.step()

