import argparse
import numpy as np
import time
import os
import pandas as pd

from TORCH_OBJECTS import *
from source.utilities import Average_Meter
from source.td_opswtw import GROUP_ENVIRONMENT
import source.MODEL__Actor.grouped_actors2 as A_Module
from source.utilities import Get_Logger, augment_data_by_8_fold
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import op_utils.instance as u_i
from source.MODEL__Actor.grouped_actors2 import multi_head_attention, reshape_by_heads


class Next_Node_Probability_Calculator_for_group(nn.Module):
    def __init__(self, embedding_dim, head_num, key_dim, logit_clipping):
        super().__init__()

        self.Wq = nn.Linear(2*embedding_dim+1, head_num * key_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * key_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * key_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * key_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.logit_clipping = logit_clipping
        self.key_dim = key_dim

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem+1, EMBEDDING_DIM)

        self.k = reshape_by_heads(
            self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(
            self.Wv(encoded_nodes), head_num=self.head_num)
        # shape = (batch, HEAD_NUM, problem+1, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.single_head_key.requires_grad = True
        # shape = (batch, EMBEDDING_DIM, problem+1)

    def forward(self, input1, input2, current_time, ninf_mask=None):
        # input1.shape = (batch, 1, EMBEDDING_DIM)
        # input2.shape = (batch, group, EMBEDDING_DIM)
        # remaining_loaded.shape = (batch, group, 1)
        # ninf_mask.shape = (batch, group, problem+1)

        with torch.no_grad():
            group_s = input2.size(1)

            #  Multi-Head Attention
            #######################################################
            input_cat = torch.cat(
                (input1.expand(-1, group_s, -1), input2, current_time), dim=2)
            # shape = (batch, group, 2*EMBEDDING_DIM+1)

            q = reshape_by_heads(self.Wq(input_cat), head_num=self.head_num)
            # shape = (batch, HEAD_NUM, group, KEY_DIM)

            out_concat = multi_head_attention(
                q, self.k, self.v, ninf_mask=ninf_mask)
            # shape = (batch, n, HEAD_NUM*KEY_DIM)

            mh_atten_out = self.multi_head_combine(out_concat)
            # shape = (batch, n, EMBEDDING_DIM)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch, n, problem+1)

        score_scaled = score / np.sqrt(self.embedding_dim)
        # shape = (batch_s, group, problem+1)

        score_clipped = self.logit_clipping * torch.tanh(score_scaled)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, group, problem+1)

        return probs


def replace_layers(actor, key_dim):
    """Function to add layers to pretrained model while retaining weights from other layers."""

    # save state dict of node_prob_calculator
    state = actor.node_prob_calculator.state_dict()

    actor_p = actor.node_prob_calculator
    # update node_prob_calculator
    actor.node_prob_calculator = Next_Node_Probability_Calculator_for_group(actor_p.embedding_dim,
                                                                            actor_p.head_num,
                                                                            key_dim,
                                                                            actor_p.logit_clipping)
    actor.node_prob_calculator.load_state_dict(state_dict=state, strict=False)

    return actor


def get_best_encoding(actor, x, adj, batch_s, seed=1, n_runs=1250, reset_actor=False):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    actor.eval()
    rwd_AM = Average_Meter()
    len_AM = Average_Meter()
    group_s = 1
    all_rwds = Tensor(np.zeros((batch_s, 0)))
    with torch.no_grad():
        for run in range(n_runs):
            env = GROUP_ENVIRONMENT(Tensor(x[np.newaxis, :, :]).expand(batch_s, -1, -1),
                                    Tensor(adj[np.newaxis, :, :]).expand(
                                        batch_s, -1, -1),
                                    deterministic=True)
            group_state, reward, done = env.reset(group_size=group_s)
            group_reward = Tensor(np.zeros((batch_s, group_s)))
            if reset_actor:
                actor.reset(group_state)

            while not done:
                action_probs = actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                # stay at depot, if you are finished
                action[group_state.finished] = 0
                group_state, reward, done = env.step(action + 1)
                group_reward += reward

            tour_cust_length = (
                group_state.selected_node_list != 1).sum(dim=2) + 1
            len_AM.push(tour_cust_length)
            mean_reward = group_reward.mean(dim=1)
            # shape = (batch_s)
            all_rwds = torch.cat((all_rwds, mean_reward[:, None]), dim=1)
            # shape = (batch_s, n_runs)
            rwd_AM.push(mean_reward)

        avg_rwd_per_encoding = all_rwds.mean(dim=1)
        # shape = (batch_s)
        max_rwd, max_el = torch.max(avg_rwd_per_encoding, dim=0)

        best_encoding = actor.node_prob_calculator.single_head_key[max_el].clone(
        ).detach()
        best_encoding.requires_grad = True
        # shape = (EMBEDDING_DIM, problem)

        best_dict = {'Encoding': best_encoding.unsqueeze(0),
                     'Nodes': actor.encoded_nodes[max_el].clone().detach().unsqueeze(0),
                     'Graph': actor.encoded_graph[max_el].clone().detach().unsqueeze(0)}

    return max_rwd, best_dict


def eval_model(actor, x, adj, batch_s, n_runs=500, seed=0, reset_actor=False):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    actor.eval()
    rwd_AM = Average_Meter()
    len_AM = Average_Meter()
    group_s = 1
    all_rwds = Tensor(np.zeros((batch_s, 0)))
    with torch.no_grad():
        for run in range(n_runs):

            env = GROUP_ENVIRONMENT(Tensor(x[np.newaxis, :, :]).expand(batch_s, -1, -1),
                                    Tensor(adj[np.newaxis, :, :]).expand(
                                        batch_s, -1, -1),
                                    deterministic=True)
            group_state, reward, done = env.reset(group_size=group_s)
            if reset_actor:
                actor.reset(group_state)

            group_reward = Tensor(np.zeros((batch_s, group_s)))

            while not done:
                action_probs = actor.get_action_probabilities(group_state)
                # shape = (batch, group, problem)
                action = action_probs.argmax(dim=2)
                # shape = (batch, group)
                # stay at depot, if you are finished
                action[group_state.finished] = 0
                group_state, reward, done = env.step(action + 1)
                group_reward += reward

            tour_cust_length = (
                group_state.selected_node_list != 1).sum(dim=2) + 1
            len_AM.push(tour_cust_length)
            mean_reward = group_reward.mean(dim=1)
            all_rwds = torch.cat((all_rwds, mean_reward[:, None]), dim=1)
            rwd_AM.push(mean_reward)  # reward was given as negative dist

    return rwd_AM.result(), len_AM.result(), all_rwds


def run_active_search(args):
    problem_size_list = [20, 50, 100, 200]
    problem_size = problem_size_list[(args.index - 1) // 250]
    inst_name = f'instance{args.index:04}'

    save_folder_name = f'ACTIVE_SEARCH_{inst_name}'
    logger, result_folder_path = Get_Logger(save_folder_name)
    logger.info(f'Active Search on {inst_name}')

    output_progress_file = os.path.join(result_folder_path, 'as_output.csv')

    hyper_params_file = os.path.join(
        args.base_load_dir, str(problem_size), 'used_HYPER_PARAMS.txt')
    hp_dict = dict()
    with open(hyper_params_file, 'r') as f:
        lines = f.read().strip().split('\n')
        for hp in lines[2:]:
            k, v = hp.split(' = ')
            hp_dict[k] = v

    base_actor = A_Module.ACTOR(embedding_dim=int(hp_dict['EMBEDDING_DIM']),
                                head_num=int(hp_dict['HEAD_NUM']),
                                logit_clipping=int(hp_dict['LOGIT_CLIPPING']),
                                encoder_layer_num=int(
                                    hp_dict['ENCODER_LAYER_NUM']),
                                ff_hidden_dim=int(hp_dict['FF_HIDDEN_DIM']),
                                key_dim=int(hp_dict['KEY_DIM'])).to(device)

    grouped_actor = A_Module.ACTOR(embedding_dim=int(hp_dict['EMBEDDING_DIM']),
                                   head_num=int(hp_dict['HEAD_NUM']),
                                   logit_clipping=int(
                                       hp_dict['LOGIT_CLIPPING']),
                                   encoder_layer_num=int(
                                       hp_dict['ENCODER_LAYER_NUM']),
                                   ff_hidden_dim=int(hp_dict['FF_HIDDEN_DIM']),
                                   key_dim=int(hp_dict['KEY_DIM'])).to(device)

    # Don't need to change base actor because it is not being trained
    grouped_actor = replace_layers(
        grouped_actor, int(hp_dict['KEY_DIM'])).to(device)

    model_path = os.path.join(args.base_load_dir, str(
        problem_size), 'ACTOR_state_dic.pt')
    base_actor.load_state_dict(torch.load(model_path, map_location=device))
    grouped_actor.load_state_dict(torch.load(model_path, map_location=device))
    grouped_actor.eval()  # Use eval mode to prevent batch normalization

    logger_start = time.time()
    timer_start = time.time()

    x, adj, _ = u_i.read_instance(os.path.join(args.data_dir, 'instances', f'{inst_name}.csv'),
                                  os.path.join(args.data_dir, 'adj-instances', f'adj-{inst_name}.csv'))
    # x.shape = (problem, 7)
    # adj.shape = (problem, problem)

    if args.evaluate_performance:
        base_rwd, base_len, base_all_rwds = eval_model(
            base_actor, x, adj, batch_s=1, reset_actor=True)
        logger.info(f'Base Model - Rwd:{base_rwd:5f}  Len:{base_len:f}')

    dist_AM = Average_Meter()
    avg_len_AM = Average_Meter()
    actor_loss_AM = Average_Meter()
    min_entropy_unique_AM = Average_Meter()
    max_entropy_unique_AM = Average_Meter()

    with torch.no_grad():
        group_s = 100
        batch_s = 120

        if args.augment:
            x_aug = augment_data_by_8_fold(Tensor(x[np.newaxis, :, :]))
            env = GROUP_ENVIRONMENT(Tensor(x_aug).repeat(
                15, 1, 1), Tensor(adj).repeat(batch_s, 1, 1))
        else:
            env = GROUP_ENVIRONMENT(Tensor(x[np.newaxis, :, :]).expand(
                batch_s, -1, -1), Tensor(adj[np.newaxis, :, :]).expand(batch_s, -1, -1))

        group_state, reward, done = env.reset(group_size=group_s)
        grouped_actor.reset(group_state)

        grouped_actor.optimizer = optim.Adam(
            [grouped_actor.node_prob_calculator.single_head_key], lr=args.lr)

    entropy_mult = Tensor(np.array(
        [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1., 3.]).repeat(8))

    if args.augment:
        base_actor.reset(GROUP_ENVIRONMENT(x_aug, Tensor(adj).repeat(8, 1, 1)))
        cur_best_enc_rwd, cur_best_dict = get_best_encoding(base_actor, x, adj, batch_s=8, seed=1,
                                                            reset_actor=False)
    else:
        cur_best_enc_rwd, cur_best_dict = get_best_encoding(base_actor, x, adj, batch_s=1, seed=1,
                                                            reset_actor=True)
    cur_best_stage = 0
    group_rwd_sum = Tensor(np.zeros((batch_s, group_s)))
    for i in range(1, args.n_steps+1):
        group_state, reward, done = env.reset(group_size=group_s)
        group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
        group_reward = Tensor(np.zeros((batch_s, group_s)))
        group_prob_list_all = Tensor(
            np.zeros((batch_s, group_s, problem_size, 0)))

        while not done:
            action_probs = grouped_actor.get_action_probabilities(group_state)
            # shape = (batch, group, problem+1)
            action = action_probs.reshape(batch_s * group_s, -1).multinomial(1) \
                .squeeze(dim=1).reshape(batch_s, group_s)
            # shape = (batch, group)
            # stay at depot, if you are finished
            action[group_state.finished] = 0
            group_state, reward, done = env.step(action + 1)
            group_reward += reward

            batch_idx_mat = torch.arange(
                batch_s)[:, None].expand(batch_s, group_s)
            group_idx_mat = torch.arange(
                group_s)[None, :].expand(batch_s, group_s)
            chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(
                batch_s, group_s)
            # shape = (batch, group)
            # done episode will gain no more probability
            chosen_action_prob[group_state.finished] = 1
            group_prob_list = torch.cat(
                (group_prob_list, chosen_action_prob[:, :, None]), dim=2)
            # shape = (batch, group, x)
            group_prob_list_all = torch.cat(
                (group_prob_list_all, action_probs[:, :, :, None]), dim=3)

        # LEARNING - Actor
        ###############################################
        group_log_prob = group_prob_list.log().sum(dim=2)
        # shape = (batch, group)
        group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)
        group_loss = -group_advantage * group_log_prob
        # shape = (batch, group)
        loss = group_loss.mean()
        group_rwd_sum += group_reward

        if 0 < i < (args.n_steps // 3):
            # set masked probability values to 0.5
            group_prob_list_all[group_prob_list_all <= 0] = 0.5
            loss_2 = (group_prob_list_all *
                      group_prob_list_all.log()).mean(dim=(1, 2, 3))
            loss += loss_2.dot(entropy_mult)

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
        min_entropy_unique_AM.push(Tensor([group_reward[0].unique().numel() / group_reward[0].numel()]),
                                   n_for_rank_0_tensor=1)
        max_entropy_unique_AM.push(Tensor([group_reward[-1].unique().numel() / group_reward[-1].numel()]),
                                   n_for_rank_0_tensor=1)
        # LOGGING
        ###############################################

        if i % 50 == 0:
            df = pd.DataFrame(columns=['Episode', 'Enc Id', 'Average Train Rwd Last 50'], data={'Episode': [i] * batch_s,
                                                                                                'Enc Id': [j for j in range(batch_s)],
                                                                                                'Average Train Rwd Last 50': group_rwd_sum.cpu().mean(dim=1) / 50})
            df.to_csv(output_progress_file, index=False, mode='a',
                      header=not os.path.isfile(output_progress_file))
            group_rwd_sum = Tensor(np.zeros((batch_s, group_s)))

        if (time.time() - logger_start > args.log_period_sec) or (i == args.n_steps):
            timestr = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - timer_start))
            log_str = 'Ep:{:07d}({:5.1f}%)  T:{:s}  ALoss:{:+5f}  Avg.dist:{:5f}  Avg.tour_len:{:f}  min entropy ' \
                      'unique:{:5f}  max entropy unique:{:5f}'.format(i, i / args.n_steps * 100, timestr,
                                                                      actor_loss_AM.result(), dist_AM.result(),
                                                                      avg_len_AM.result(),
                                                                      min_entropy_unique_AM.result(),
                                                                      max_entropy_unique_AM.result())
            logger.info(log_str)
            logger_start = time.time()

        if i in [500, 750, 1000, 1250]:
            best_enc_rwd, best_dict = get_best_encoding(
                grouped_actor, x, adj, batch_s=batch_s, seed=1)
            if best_enc_rwd > cur_best_enc_rwd:
                cur_best_enc_rwd = best_enc_rwd
                cur_best_dict = best_dict
                cur_best_stage = i

    best_enc_rwd, best_dict = get_best_encoding(
        grouped_actor, x, adj, batch_s=batch_s, seed=1)

    if best_enc_rwd > cur_best_enc_rwd:
        cur_best_enc_rwd = best_enc_rwd
        cur_best_dict = best_dict
        cur_best_stage = args.n_steps

    #     if args.evaluate_performance and cur_best_stage != 0:
    #         new_env = GROUP_ENVIRONMENT(Tensor(x[np.newaxis, :, :]), Tensor(adj[np.newaxis, :, :]))
    #         group_state, reward, done = new_env.reset(group_size=1)
    #         # Create new actor
    #         new_actor = A_Module.ACTOR(embedding_dim=int(hp_dict['EMBEDDING_DIM']),
    #                                        head_num=int(hp_dict['HEAD_NUM']),
    #                                        logit_clipping=int(hp_dict['LOGIT_CLIPPING']),
    #                                        encoder_layer_num=int(hp_dict['ENCODER_LAYER_NUM']),
    #                                        ff_hidden_dim=int(hp_dict['FF_HIDDEN_DIM']),
    #                                        key_dim=int(hp_dict['KEY_DIM'])).to(device)
    #         new_actor.load_state_dict(torch.load(model_path, map_location=device))
    #         new_actor.eval()
    #         new_actor.reset(group_state)
    #         new_actor.node_prob_calculator.single_head_key = best_encoding
    #
    #         final_rwd, final_len, final_all_rwds = eval_model(new_actor, x, adj, batch_s=1, reset_actor=False)
    #
    # else:
    #
    #     if args.evaluate_performance:
    #         logger.info(f'Final Model - Rwd:{cur_best_enc_rwd:5f}')

    logger.info(f'Best Encoding Reward: {cur_best_enc_rwd}')
    logger.info(f'Best Encoding Found after: {cur_best_stage} train steps.')

    encoding_save_path = os.path.join(
        args.encoding_save_dir, f'{inst_name}_encoding.pt')
    torch.save(cur_best_dict, encoding_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Active Search on Instance')
    parser.add_argument(
        'index', type=int, help='Index of instance to run active search on (uses 1-indexing)')
    parser.add_argument('--base_load_dir', default='./models/base',
                        help='Dir where base models are stored. Defaults to "./models/base/"')
    parser.add_argument('--encoding_save_dir', default='./models/encodings',
                        help='Root directory to store best encodings. Defaults to "./models/encodings"')
    parser.add_argument('--data_dir', default='./data/test',
                        help='Dir where data is stored. Defaults to "./data/test"')
    parser.add_argument('--n_steps', type=int, default=1500,
                        help='Number of train steps to run. Defaults to 500')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Actor learning rate. Defaults to 1e-2')
    parser.add_argument('--augment', action='store_true',
                        help='Perform 8x augmentation to create 8 different starting encodings')
    parser.add_argument('--log_period_sec', type=int, default=15,
                        help='Number of seconds between logs. Defaults to 15')
    parser.add_argument('--evaluate_performance', action='store_true',
                        help='Runs base and final model on val dataset')
    run_args = parser.parse_args()

    run_active_search(run_args)
