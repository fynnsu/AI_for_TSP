
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


import logging
import os
import datetime
import pytz
import pandas as pd
import json
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from op_utils import instance as u_i

import numpy as np

from TORCH_OBJECTS import *


########################################
# Get_Logger
########################################
tz = pytz.timezone('Europe/Berlin')


def timetz(*args):
    return datetime.datetime.now(tz).timetuple()


def Get_Logger(SAVE_FOLDER_NAME):
    # make_dir
    #######################################################
    prefix = datetime.datetime.now(pytz.timezone('Europe/Berlin')).strftime("%Y%m%d_%H%M__")
    result_folder_no_postfix = "./result/{}".format(prefix + SAVE_FOLDER_NAME)

    result_folder_path = result_folder_no_postfix
    folder_idx = 0
    while os.path.exists(result_folder_path):
        folder_idx += 1
        result_folder_path = result_folder_no_postfix + "({})".format(folder_idx)

    os.makedirs(result_folder_path)

    # Logger
    #######################################################
    logger = logging.getLogger(result_folder_path)  # this already includes streamHandler??

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('{}/log.txt'.format(result_folder_path))

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    formatter.converter = timetz

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.INFO)

    return logger, result_folder_path


def Extract_from_LogFile(result_folder_path, variable_name):
    logfile_path = '{}/log.txt'.format(result_folder_path)
    with open(logfile_path) as f:
        datafile = f.readlines()
    found = False  # This isn't really necessary
    for line in reversed(datafile):
        if variable_name in line:
            found = True
            m = re.search(variable_name + '[^\n]+', line)
            break
    exec_command = "Print(No such variable found !!)"
    if found:
        return m.group(0)
    else:
        return exec_command


########################################
# Average_Meter
########################################

class Average_Meter:
 
    def __init__(self):
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.sum = torch.tensor(0.).to(device)
        self.count = 0

    def push(self, some_tensor, n_for_rank_0_tensor=None):
        assert not some_tensor.requires_grad # You get Memory error, if you keep tensors with grad history
        
        rank = len(some_tensor.shape)

        if rank == 0: # assuming "already averaged" Tensor was pushed
            self.sum += some_tensor * n_for_rank_0_tensor
            self.count += n_for_rank_0_tensor
            
        else:
            self.sum += some_tensor.sum()
            self.count += some_tensor.numel()

    def peek(self):
        average = (self.sum / self.count).tolist()
        return average

    def result(self):
        average = (self.sum / self.count).tolist()
        self.reset()
        return average





########################################
# View NN Parameters
########################################

def get_n_params1(model):
    pp = 0
    for p in list(model.parameters()):
        nn_count = 1
        for s in list(p.size()):
            nn_count = nn_count * s
        pp += nn_count
        print(nn_count)
        print(p.shape)
    print("Total: {:d}".format(pp))


def get_n_params2(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


def get_n_params3(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_structure(model):
    print(model)




########################################
# Augment xy data
########################################

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape = (batch_s, problem, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape = (batch, problem, 1)
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1-x, y), dim=2)
    dat3 = torch.cat((x, 1-y), dim=2)
    dat4 = torch.cat((1-x, 1-y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1-y, x), dim=2)
    dat7 = torch.cat((y, 1-x), dim=2)
    dat8 = torch.cat((1-y, 1-x), dim=2)

    data_augmented = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape = (8*batch, problem, 2)

    return data_augmented


def augment_data_by_8_fold(data):
    # data.shape = (batch, problem, 7)
    # data attributes = [1-index, x [0 to 200), y [0 to 50), lower tw, upper tw, prize, max_time]
    x = data[:, :, [1]]
    y = data[:, :, [2]]
    index = data[:, :, [0]]
    const = data[:, :, 3:]
    data2 = torch.cat((index, 199-x, y, const), dim=2)
    data3 = torch.cat((index, x, 49-y, const), dim=2)
    data4 = torch.cat((index, 199-x, 49-y, const), dim=2)
    data5 = torch.cat((index, y*4, x/4, const), dim=2)
    data6 = torch.cat((index, (49-y)*4, x/4, const), dim=2)
    data7 = torch.cat((index, y*4, (199-x)/4, const), dim=2)
    data8 = torch.cat((index, (49-y)*4, (199-x)/4, const), dim=2)

    data_augmented = torch.cat((data, data2, data3, data4, data5, data6, data7, data8), dim=0)

    return data_augmented

########################################
# Convert Instance Files to grouped dataset
########################################


def instances_to_np_dataset(data_dir):
    instances = os.listdir(os.path.join(data_dir, 'instances'))
    data = []
    adjs = []
    for inst in tqdm(instances):
        x, adj, _ = u_i.read_instance(os.path.join(data_dir, 'instances', inst),
                                      os.path.join(data_dir, 'adjs', f'adj-{inst}'))
        data.append(x)
        adjs.append(adj)

    return np.array(data).astype(np.float32), np.array(adjs).astype(np.float32)

def plot_training_curves(run_names, run_dir_names, shape):
    dfs = []
    for name, dir_name in zip(run_names, run_dir_names):
        df = pd.read_csv(os.path.join('td_opswtw/result/', dir_name, 'train_output.csv'), index_col='Epoch')
        dfs.append(df)
        plt.plot(df.index, df['Val rwd'], label=name)

    plt.legend()
    plt.title('Val Reward')
    plt.savefig('td_opswtw/plots/Val_rwd.png')
    plt.show()

    for name, df in zip(run_names, dfs):
        plt.plot(df.index, df['Val len'], label=name)

    plt.legend()
    plt.title('Val Length')
    plt.savefig('td_opswtw/plots/Val_len.png')
    plt.show()

    fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1])

    for i in range(len(dfs)):
        ax[i // shape[1]][i % shape[1]].plot(dfs[i]['Val rwd'], label='Validation')
        ax[i // shape[1]][i % shape[1]].plot(dfs[i]['Train rwd'], label='Train')
        ax[i // shape[1]][i % shape[1]].title.set_text(run_names[i])
        ax[i // shape[1]][i % shape[1]].legend()

    fig.suptitle('Reward')
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.savefig('td_opswtw/plots/Train_v_val_rwd.png')
    plt.show()

    fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1])

    for i in range(len(dfs)):
        ax[i // shape[1]][i % shape[1]].plot(dfs[i]['Val len'], label='Validation')
        ax[i // shape[1]][i % shape[1]].plot(dfs[i]['Train len'], label='Train')
        ax[i // shape[1]][i % shape[1]].title.set_text(run_names[i])
        ax[i // shape[1]][i % shape[1]].legend()

    fig.suptitle('Tour Length')
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.savefig('td_opswtw/plots/Train_v_val_len.png')
    plt.show()







########################################
# Plot Tour
########################################

def plot_tour(data, tour):
    # data.shape = (problem, 7)
    # tour.shape = (1-dim up to problem)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(*zip(*data[:, 1:3]), 'ko')
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 50)
    start_node = tour[0]
    start_pos = data[start_node - 1, 1:3]
    ax.plot(*start_pos, 'ro')
    ax.text(start_pos[0] * (1 + 0.01), start_pos[1] * (1 + 0.01), 1, fontsize=8)
    distance = 0.
    N = len(tour)
    for i in range(1, N):
        start_pos = data[start_node-1, 1:3]
        next_node = tour[i]
        end_pos = data[next_node-1, 1:3]
        # ax.plot(*end_pos, 'ko')
        ax.text(end_pos[0] * (1 + 0.01), end_pos[1] * (1 + 0.01), int(next_node), fontsize=8)
        ax.annotate("",
                xy=end_pos, xycoords='data',
                xytext=start_pos, textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3"))

        distance += np.linalg.norm(end_pos - start_pos)
        start_node = next_node

    ax.plot(*data[start_node - 1, 1:3], 'bo')
    textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, # Textbox
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()


def plot_tour2(group_env, save_dir, index=0):
    """Plot first tour in each group to file"""
    data = group_env.group_state.data
    plt.plot(data[index, :, 1], data[index, :, 2], 'ko')

    plt.xlabel('x')
    plt.ylabel('y')

    tour = group_env.group_state.selected_node_list
    tour_times = group_env.group_state.tour_time_list
    start_node = int(tour[index, 0, 0])
    for i in range(1, len(tour[index, 0])):
        end_node = int(tour[index, 0, i])
        if tour_times[index, 0, i] > data[index, end_node-1, 3]:
            if tour_times[index, 0, i] < data[index, end_node-1, 4]:
                plt.plot(data[index, end_node-1, 1], data[index, end_node-1, 2], 'go')
            else:
                plt.plot(data[index, end_node-1, 1], data[index, end_node-1, 2], 'ro')
        else:
            plt.plot(data[index, end_node-1, 1], data[index, end_node-1, 2], 'yo')
        plt.plot(data[index, [start_node-1, end_node-1], 1],
                 data[index, [start_node-1, end_node-1], 2],
                 color='k')
        start_node = end_node

    bad_customers = data[index, :, 3].gt(data[index, :, 6])
    plt.plot(data[index, bad_customers, 1],
             data[index, bad_customers, 2], 'mo')

    plt.plot(data[index, 0, 1], data[index, 0, 2], 'bo')

    plt.figure(figsize=(10, 10))
    plt.show()


def plot_group(data, group_tours):
    # data.shape = (problem, 7)
    # group_tours.shape = (group, problem+1)

    problem_s = data.shape[0]

    # plot points
    plt.plot(data[:, 1], data[:, 2], 'ko')

    plt.xlabel('x')
    plt.ylabel('y')

    adj_freq = np.zeros((problem_s, problem_s))

    for sample in group_tours:
        prev_node = 1
        for node in sample[1:]:
            adj_freq[prev_node-1][node-1] += 1
            prev_node = node
            if prev_node == 1:
                break

    max_freq = np.max(adj_freq)

    for i in range(problem_s):
        for j in range(problem_s):
            plt.plot(data[[i, j], 1], data[[i,j], 2], color='r', alpha=adj_freq[i,j] / max_freq)

    bad_customers = data[:, 3] > data[:, 6]
    plt.plot(data[bad_customers, 1],
             data[bad_customers, 2], 'mo')
    plt.plot(data[0, 1], data[0, 2], 'bo')
    plt.figure(figsize=(10, 10))
    plt.show()


def read_json_tours(file_path):
    f = open(file_path)
    submission = json.load(f)

    instances = []
    sub_keys = sorted(submission.keys())
    for inst_name in sub_keys:
        instances.append(np.array(list(submission[inst_name]['tours'].values())))

    return instances


def tour_group_intersect_over_union(tours):
    # tours.shape = (group, problem+1)
    problem_s = np.max(tours)
    inter = set([i for i in range(1, problem_s+1)])
    union = set()
    for tour in tours:
        t_set = {1}
        for node in tour[1:]:
            if node == 1:
                break

            t_set.add(node)

        inter = inter.intersection(t_set)
        union = union.union(t_set)

    return len(inter) /  len(union)

if __name__ == '__main__':
    from source.td_opswtw import DATA_LOADER__FROM_FILE
    from TORCH_OBJECTS import *
    data_loader = DATA_LOADER__FROM_FILE('../data/train/20/instances', '../data/train/20/adjs', 5)
    for data, adj in data_loader:
        break

    tour = LongTensor(np.arange(20)+1)
    plot_tour(data[0], tour[:10])