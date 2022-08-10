
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
import pandas as pd
import os
import time

# For debugging

from TORCH_OBJECTS import *

from source.utilities import Average_Meter
from source.td_opswtw import DATA_LOADER__FROM_NPY, GROUP_ENVIRONMENT, DATA_LOADER__FROM_FILE


########################################
# EVAL
########################################

eval_result = []

def eval_dataset(actor, data_loader, args):
    rwd_AM = Average_Meter()
    len_AM = Average_Meter()

    with torch.no_grad():
        for xs, adjs in data_loader:
            # xs.shape = (batch, problem, 7)
            # adjs.shape = (batch, problem, problem)

            batch_s = xs.size(0)

            env = GROUP_ENVIRONMENT(xs, adjs, max_time_pen_mult=args.max_time_pen_mult,
                                    dist=args.dist)
            group_s = 1
            group_state, reward, done = env.reset(group_size=group_s)
            actor.reset(group_state)

            group_reward = Tensor(np.zeros((batch_s, group_s)))

            while not done:
                action_probs = actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action+1)
                group_reward += reward


            tour_cust_length = (group_state.selected_node_list != 1).sum(dim=2) + 1
            len_AM.push(tour_cust_length)
            mean_reward = group_reward.mean(dim=1)
            max_reward, _ = group_reward.max(dim=1)
            rwd_AM.push(mean_reward)  # reward was given as negative dist

    rwd_avg = rwd_AM.result()
    len_avg = len_AM.result()

    return rwd_avg, len_avg


def EVAL(grouped_actor, args, train_progress_file, epoch, timer_start, logger):
    global eval_result

    grouped_actor.eval()

    # test_loader = DATA_LOADER__RANDOM(num_sample=TEST_DATASET_SIZE,
    #                                        num_nodes=PROBLEM_SIZE,
    #                                        batch_size=args.test_batch_size)

    test_loader = None
    if args.dataset_mode == 0:
        test_loader = DATA_LOADER__FROM_NPY(data_dir=args.test_data_dir,
                                             batch_size=args.test_batch_size)
        train_loader = DATA_LOADER__FROM_NPY(data_dir=args.train_data_dir,
                                             batch_size=args.test_batch_size,
                                             n_instances=len(test_loader.dataset))
    elif args.dataset_mode == 1:
        test_loader = DATA_LOADER__FROM_FILE(data_dir=os.path.join(args.test_data_dir, 'instances/'),
                                              adj_dir=os.path.join(args.test_data_dir, 'adjs/'),
                                              batch_size=args.test_batch_size)
        # TODO: Fix train data size for this mode
        train_loader = DATA_LOADER__FROM_FILE(data_dir=os.path.join(args.train_data_dir, 'instances/'),
                                              adj_dir=os.path.join(args.train_data_dir, 'adjs/'),
                                              batch_size=args.test_batch_size)
    else:
        raise NotImplemented("Other Modes not yet implemented")
        # TODO: Fix Random Loader

    val_rwd_avg, val_len_avg = eval_dataset(grouped_actor, test_loader, args)
    train_rwd_avg, train_len_avg = eval_dataset(grouped_actor, train_loader, args)

    eval_result.append(val_rwd_avg)

    # Pandas Train log
    df = pd.DataFrame(columns=['Epoch', 'Time', 'Train rwd', 'Train len', 'Val rwd', 'Val len'])
    timestr = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
    df = df.append({'Epoch': epoch,
               'Time': timestr,
               'Train rwd': train_rwd_avg,
               'Train len': train_len_avg,
               'Val rwd': val_rwd_avg,
               'Val len': val_len_avg},
                   ignore_index=True)
    df.to_csv(train_progress_file, index=False, mode='a',header=not os.path.isfile(train_progress_file))

    logger.info('--------------------------------------------------------------------------')
    logger.info('<<< EVAL after Epoch:{:03d} >>>'.format(epoch))
    logger.info('Validation:  Avg.rwd:{:f}  Avg.tour_len:{:f}'.format(val_rwd_avg, val_len_avg))
    logger.info('Train:  Avg.rwd:{:f}  Avg.tour_len:{:f}'.format(train_rwd_avg, train_len_avg))
    logger.info('--------------------------------------------------------------------------')
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')
    logger.info('--------------------------------------------------------------------------')

