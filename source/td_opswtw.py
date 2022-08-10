
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

####################################
# EXTERNAL LIBRARY
####################################
import random
import torch.random

import numpy as np
import os
import op_utils.instance as u_i

# For debugging
from IPython.core.debugger import set_trace

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


####################################
# PROJECT VARIABLES
####################################
from TORCH_OBJECTS import *

####################################
# POMO STUFF BELOW
# ##################################


####################################
# DATA
####################################
def DATA_LOADER__RANDOM(num_sample, num_nodes, batch_size):
    dataset = Dataset__Random(num_sample=num_sample, num_nodes=num_nodes)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_fn)
    return data_loader


class Dataset__Random(Dataset):
    def __init__(self, num_sample, num_nodes):
        self.num_sample = num_sample
        self.num_nodes = num_nodes

    def __getitem__(self, index):
        x, adj = u_i.make_instance(self.num_nodes)
        return x.astype(np.float32), adj.astype(np.float32)

    def __len__(self):
        return self.num_sample


def DATA_LOADER__FROM_FILE(data_dir, adj_dir, batch_size):
    dataset = Dataset__FromFile(data_dir, adj_dir)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=collate_fn)
    return data_loader


class Dataset__FromFile(Dataset):
    def __init__(self, data_dir, adj_dir):
        self.data_dir = data_dir
        self.adj_dir = adj_dir
        self.instances = os.listdir(self.data_dir)
        self.num_sample = len(self.instances)

    def __getitem__(self, index):
        x, adj, _ = u_i.read_instance(os.path.join(self.data_dir, self.instances[index]),
                                      os.path.join(self.adj_dir, f'adj-{self.instances[index]}'))

        return x.astype(np.float32), adj.astype(np.float32)

    def __len__(self):
        return len(self.instances)


def DATA_LOADER__FROM_NPY(data_dir, batch_size, n_instances=None):
    dataset = Dataset__FromNPY(data_dir, n_instances)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=collate_fn)
    return data_loader


class Dataset__FromNPY(Dataset):
    def __init__(self, data_dir, n_instances):
        self.data_dir = data_dir
        self.instance_file = os.path.join(data_dir, 'instances.npy')
        if not os.path.isfile(self.instance_file):
            raise FileNotFoundError(
                'instances.npy file not found in location provided: ' + self.instance_file)
        self.adj_file = os.path.join(data_dir, 'adjs.npy')
        if not os.path.isfile(self.adj_file):
            raise FileNotFoundError(
                'adjs.npy file not found in data_dir provided.')

        self.x = np.load(self.instance_file)
        self.adj = np.load(self.adj_file)
        self.n_instances = self.x.shape[0]

        # if requested dataset size is smaller than true dataset size, take a random subset
        if n_instances and n_instances < self.x.shape[0]:
            indices = np.random.choice(
                self.x.shape[0], n_instances, replace=False)
            self.x = self.x[indices, :, :]
            self.adj = self.adj[indices, :, :]
            self.n_instances = n_instances

    def __getitem__(self, index):
        return self.x[index, :, :], self.adj[index, :, :]

    def __len__(self):
        return self.n_instances


def collate_fn(batch):
    xs, adjs = zip(*batch)
    return Tensor(xs), Tensor(adjs)


####################################
# STATE
####################################
class GROUP_STATE:
    # Part of GROUP_ENVIRONMENT found below

    def create_from_existing(self, group_state):
        # Used to create replicas of an environment state, to perform MC rollouts on
        assert group_state.group_s == 1 and group_state.batch_s == 1, "The base group and batch size must both be 1"
        # Assume input has batch_s=1, group_s=1
        self.data = group_state.data.repeat(self.batch_s, 1, 1)
        self.adj = group_state.adj.repeat(self.batch_s, 1, 1)
        if self.deterministic:
            gen = torch.Generator(device=device)
            self.adj = self.adj * \
                torch.randint(1, 101, size=self.adj.shape,
                              device=device, generator=gen) / 100

        self.problem_s = self.data.size(1)
        self.dist = 'normal'
        self.tw_pen = -1
        self.maxT_pen = -1
        self.max_times = self.data[:, 0, 6].unsqueeze(
            dim=1).expand(-1, self.group_s)

        self.selected_count = group_state.selected_count
        self.current_nodes = group_state.current_nodes.repeat(
            self.batch_s, self.group_s)
        # shape = (batch, group)
        self.selected_node_list = group_state.selected_node_list.repeat(
            self.batch_s, self.group_s, 1)
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.returned_to_depot = group_state.returned_to_depot.repeat(
            self.batch_s, self.group_s)
        # shape = (batch, group)

        self.tour_time = group_state.tour_time.repeat(
            self.batch_s, self.group_s)
        # shape = (batch, group)
        self.tour_time_list = group_state.tour_time_list.repeat(
            self.batch_s, self.group_s, 1)
        # shape = (batch, group, selected_count)

        self.pen = group_state.pen.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)
        self.rewards = group_state.rewards.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)
        self.feas = group_state.feas.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)
        self.time_t = group_state.time_t.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)
        self.rwd_t = group_state.rwd_t.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)
        self.pen_t = group_state.pen_t.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)
        self.violation_t = group_state.violation_t.repeat(
            self.batch_s, self.group_s)
        # shape = (batch, group)

        # self.loaded = Tensor(np.ones((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.visited_ninf_flag = group_state.visited_ninf_flag.repeat(
            self.batch_s, self.group_s, 1)
        # shape = (batch, group, problem)
        self.ninf_mask = group_state.ninf_mask.repeat(
            self.batch_s, self.group_s, 1)
        # shape = (batch, group, problem)
        self.finished = group_state.finished.repeat(self.batch_s, self.group_s)
        # shape = (batch, group)

        self.bad_customer_ninf_mask = group_state.bad_customer_ninf_mask.repeat(
            self.batch_s, self.group_s, 1)
        # shape = (batch, group, problem)

    def __init__(self, data=None, adj=None, group_state=None, batch_s=None, group_s=1, deterministic=False, use_expected=False,
                 max_time_pen_mult=-1.,
                 dist='normal'):
        # data.shape = (batch, problem, 7)

        if data is None and adj is None and group_state is not None:
            self.batch_s = batch_s
            self.group_s = group_s
            self.deterministic = deterministic
            self.create_from_existing(group_state)
            return

        self.deterministic = deterministic
        self.dist = dist

        self.batch_s = data.size(0)
        self.problem_s = data.size(1)
        self.group_s = group_s
        # [1-index, x, y, lower tw, upper tw, prize, max_time]
        self.data = data
        # shape = (batch, problem, 7)

        # Adjacency Matrix
        if use_expected:  # Modified environment with all travel taking the expected time instead of a stochastic value
            self.adj = adj * 0.505
            self.deterministic = True
        elif deterministic:  # Deterministic environment with random travel times
            gen = torch.Generator(device=device)
            gen.seed()
            self.adj = adj * \
                torch.randint(1, 101, size=adj.shape,
                              device=device, generator=gen) / 100
        else:
            self.adj = adj
        # shape = (batch, problem, problem)

        # Constants
        ####################################
        self.tw_pen = -1
        self.maxT_pen = max_time_pen_mult
        self.max_times = self.data[:, 0, 6].unsqueeze(
            dim=1).expand(-1, self.group_s)
        # shape = (batch, group)

        # History
        ####################################
        self.selected_count = 1
        # Nodes are 1-indexed (start at depot)
        self.current_nodes = LongTensor(np.ones((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.selected_node_list = LongTensor(
            np.zeros((self.batch_s, self.group_s, 0)))
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_nodes[:, :, None]), dim=2)
        # shape = (batch, group, selected_count)

        # Status
        ####################################
        self.returned_to_depot = BoolTensor(
            np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)

        self.tour_time = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.tour_time_list = Tensor(np.zeros((self.batch_s, self.group_s, 1)))
        # shape = (batch, group, selected_count)

        self.pen = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.rewards = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.feas = BoolTensor(
            np.ones((self.batch_s, self.group_s)))  # feasible
        # shape = (batch, group)
        self.time_t = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.rwd_t = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.pen_t = Tensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.violation_t = LongTensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)

        # self.loaded = Tensor(np.ones((self.batch_s, self.group_s)))
        # shape = (batch, group)
        self.visited_ninf_flag = Tensor(
            np.zeros((self.batch_s, self.group_s, self.problem_s)))
        # shape = (batch, group, problem)
        self.ninf_mask = Tensor(
            np.zeros((self.batch_s, self.group_s, self.problem_s)))
        # shape = (batch, group, problem)
        self.finished = BoolTensor(np.zeros((self.batch_s, self.group_s)))
        # shape = (batch, group)

        self.bad_customer_ninf_mask = Tensor(
            np.zeros((self.batch_s, self.group_s, self.problem_s)))
        bc_mask = self.data[:, :, 3].gt(self.data[:, :, 6])
        self.bad_customer_ninf_mask[bc_mask[:, None, :].expand(
            self.batch_s, self.group_s, self.problem_s)] = -np.inf
        # shape = (batch, group, problem)
        self.ninf_mask += self.bad_customer_ninf_mask

    def expand_group(self, factor):
        assert type(
            factor) == int and factor > 0, "Factor must be a positive integer"

        self.group_s = self.group_s * factor
        # shape change: group -> group * factor

        self.max_times = self.max_times.repeat_interleave(factor, dim=1)

        self.current_nodes = self.current_nodes.repeat_interleave(
            factor, dim=1)
        self.selected_node_list = self.selected_node_list.repeat_interleave(
            factor, dim=1)

        self.returned_to_depot = self.returned_to_depot.repeat_interleave(
            factor, dim=1)
        self.tour_time = self.tour_time.repeat_interleave(factor, dim=1)
        self.tour_time_list = self.tour_time_list.repeat_interleave(
            factor, dim=1)
        self.pen = self.pen.repeat_interleave(factor, dim=1)
        self.rewards = self.rewards.repeat_interleave(factor, dim=1)
        self.feas = self.feas.repeat_interleave(factor, dim=1)
        self.time_t = self.time_t.repeat_interleave(factor, dim=1)
        self.rwd_t = self.rwd_t.repeat_interleave(factor, dim=1)
        self.pen_t = self.pen_t.repeat_interleave(factor, dim=1)
        self.finished = self.finished.repeat_interleave(factor, dim=1)
        self.violation_t = self.violation_t.repeat_interleave(factor, dim=1)

        self.visited_ninf_flag = self.visited_ninf_flag.repeat_interleave(
            factor, dim=1)
        self.ninf_mask = self.ninf_mask.repeat_interleave(factor, dim=1)
        self.bad_customer_ninf_mask = self.bad_customer_ninf_mask.repeat_interleave(
            factor, dim=1)

    def move_to(self, selected_idx_mat, noise_clump_size=1, rng=None):
        # selected_idx_mat.shape = (batch, group)
        assert self.group_s % noise_clump_size == 0, "group must be divisible by noise_clump_size"

        # Reward and Penalty for current time step
        rwd_t = Tensor(np.zeros((self.batch_s, self.group_s)))
        pen_t = Tensor(np.zeros((self.batch_s, self.group_s)))
        self.violation_t = LongTensor(np.zeros((self.batch_s, self.group_s)))

        # If returned to the depot, keep moving to the depot
        selected_idx_mat.masked_fill_(self.returned_to_depot, 1)

        previous_tour_times = self.tour_time.detach().clone()

        # General gathering index
        gathering_index = selected_idx_mat.unsqueeze(dim=2)

        adj_group = self.adj.unsqueeze(1).expand(-1, self.group_s, -1, -1)
        # shape = (batch, group, problem, problem)
        cur_gather_index = self.current_nodes[:, :,
                                              None, None].expand(-1, -1, self.problem_s, -1)
        # shape = (batch, group, problem, 1)
        cur_adj_group = adj_group.gather(
            dim=3, index=cur_gather_index-1).squeeze(dim=3)
        # shape = (batch, group, problem)
        # shape = (batch, group, 1)
        travel_times = cur_adj_group.gather(
            dim=2, index=gathering_index-1).squeeze(dim=2)
        # shape = (batch, group)
        if self.deterministic:
            self.tour_time += travel_times
        else:
            # Allows multiple rollouts in a group to share a noise sample
            noise_s = self.group_s // noise_clump_size
            # value (used when training with forced first steps)
            if rng is not None:
                if self.dist == 'normal':
                    # Use separate random number generator (required to maintain rng state for tour generation when other
                    # calls to random number generators are used at the same time)
                    noise = Tensor(rng.randint(
                        1, 101, size=(self.batch_s, noise_s)) / 100)
                elif self.dist == 'rayleigh':
                    noise = Tensor(rng.rayleigh(
                        size=(self.batch_s, noise_s)) / 2)
                else:
                    raise ValueError(f'Invalid Distribution name: {self.dist}')
            else:
                if self.dist == 'normal':
                    noise = torch.randint(1, 101, size=(
                        self.batch_s, noise_s), device=device) / 100
                elif self.dist == 'rayleigh':
                    noise = Tensor(np.random.rayleigh(
                        size=(self.batch_s, noise_s)) / 2)
                else:
                    raise ValueError(f'Invalid Distribution name: {self.dist}')
            self.tour_time += noise.repeat_interleave(
                noise_clump_size, dim=1) * travel_times
        # History
        ####################################
        self.selected_count += 1
        self.current_nodes = selected_idx_mat
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2)
        self.tour_time_list = torch.cat(
            (self.tour_time_list, self.tour_time[:, :, None]), dim=2)

        # Status
        ####################################
        tw_list = self.data[:, None, :, 3:5].expand(-1, self.group_s, -1, -1)
        # shape = (batch, group, problem, 2)
        selected_tw = tw_list.gather(dim=2, index=gathering_index.unsqueeze(
            dim=3).expand(-1, -1, -1, 2)-1).squeeze(dim=2)
        # shape = (batch, group, 2)
        tw_lower_mask = self.tour_time.lt(selected_tw[:, :, 0]).logical_and(
            self.returned_to_depot.logical_not())
        # shape = (batch, group)
        tw_upper_mask = self.tour_time.gt(selected_tw[:, :, 1]).logical_and(
            self.returned_to_depot.logical_not())
        # shape = (batch, group)

        self.feas.masked_fill_(tw_upper_mask, False)
        self.violation_t.masked_fill_(tw_upper_mask, 1)

        pen_t[tw_upper_mask] += self.tw_pen

        self.tour_time[tw_lower_mask] = selected_tw[:, :,
                                                    0][tw_lower_mask]  # Set time to lower time window
        self.time_t = self.tour_time - previous_tour_times

        tw_valid_mask = self.tour_time.le(selected_tw[:, :, 1]).logical_and(
            self.returned_to_depot.logical_not())

        prize_list = self.data[:, None, :, 5].expand(
            self.batch_s, self.group_s, -1)
        # shape = (batch, group, problem)
        rwd_t[tw_valid_mask] = prize_list.gather(
            dim=2, index=gathering_index-1).squeeze(dim=2)[tw_valid_mask]
        self.rewards += rwd_t

        max_time_mask = self.tour_time.gt(self.max_times).logical_and(
            self.returned_to_depot.logical_not())
        pen_t[max_time_mask] += self.maxT_pen * self.problem_s
        self.pen += pen_t
        self.feas[max_time_mask] = False
        self.violation_t[max_time_mask] = 2
        self.rwd_t = rwd_t
        self.pen_t = pen_t

        self.returned_to_depot = selected_idx_mat == 1

        batch_idx_mat = torch.arange(self.batch_s)[:, None].expand(
            self.batch_s, self.group_s)
        group_idx_mat = torch.arange(self.group_s)[None, :].expand(
            self.batch_s, self.group_s)
        self.visited_ninf_flag[batch_idx_mat,
                               group_idx_mat, selected_idx_mat-1] = -np.inf
        self.finished = self.returned_to_depot
        # shape = (batch, group)

        # Status Edit
        ####################################
        closed_customers = self.tour_time.unsqueeze(
            dim=2).gt(tw_list[:, :, :, 1])
        # shape = (batch, group, problem)

        self.ninf_mask = self.visited_ninf_flag.clone()
        # shape = (batch, group, problem)

        # Customers with lower time window > max time
        self.ninf_mask += self.bad_customer_ninf_mask

        # Mask nodes with exceeded time windows
        self.ninf_mask[closed_customers] = -np.inf

        # Mask all nodes when max time is exceeded
        self.ninf_mask[max_time_mask[:, :, None].expand(
            self.batch_s, self.group_s, self.problem_s)] = -np.inf
        self.ninf_mask[:, :, 0] = 0  # Unmask Depot Node

        self.ninf_mask[self.finished[:, :, None].expand(
            self.batch_s, self.group_s, self.problem_s)] = 0
        # do not mask finished episode

        return rwd_t, pen_t


####################################
# ENVIRONMENT
####################################
class GROUP_ENVIRONMENT:
    def __init__(self, xs=None, adjs=None, group_env=None, batch_s=None, group_s=None, seed=None, deterministic=False,
                 use_expected=False,
                 max_time_pen_mult=-1.,
                 dist='normal'):
        # xs.shape = (batch, problem, 7)
        # adjs.shape = (batch, problem, problem)

        # deterministic causes the travel times to be calculated once, when the environment is reset
        # and then all moves use the adj matrix for travel times

        if seed:
            torch.random.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Copy constructor
        if xs is None and adjs is None and group_env is not None:
            self.batch_s = batch_s
            self.group_s = group_s
            self.group_state = GROUP_STATE(group_state=group_env.group_state, batch_s=batch_s, group_s=group_s,
                                           deterministic=deterministic,
                                           max_time_pen_mult=max_time_pen_mult,
                                           dist=dist)
            self.xs = self.group_state.data
            self.adj = self.group_state.adj
            self.deterministic = deterministic
            self.problem_s = self.xs.size(0)
            self.use_expected = use_expected
            if use_expected:
                self.deterministic = True
        else:
            self.problem_s = xs.size(1)
            self.batch_s = xs.size(0)
            self.group_s = None
            self.group_state = None
            self.data = xs
            self.adj = adjs
            self.deterministic = deterministic
            self.use_expected = use_expected
            if use_expected:
                self.deterministic = True

        self.dist = dist
        self.max_time_pen_mult = max_time_pen_mult

    def expand_group(self, factor):
        assert type(
            factor) == int and factor > 0, "Factor must be a positive integer"
        self.group_s = self.group_s * factor

        if self.group_state is not None:
            self.group_state.expand_group(factor)

        return self.group_state

    def reset(self, group_size):
        self.group_s = group_size
        self.group_state = GROUP_STATE(data=self.data,
                                       adj=self.adj,
                                       group_s=group_size,
                                       deterministic=self.deterministic,
                                       use_expected=self.use_expected,
                                       max_time_pen_mult=self.max_time_pen_mult,
                                       dist=self.dist)

        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat, noise_clump_size=1, rng=None):
        # selected_idx_mat.shape = (batch, group)

        # move state
        reward, penalty = self.group_state.move_to(selected_idx_mat, noise_clump_size=noise_clump_size,
                                                   rng=rng)

        done = self.group_state.finished.all()
        # state.finished.shape = (batch, group)
        return self.group_state, (reward + penalty), done
