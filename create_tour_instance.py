import numpy as np
import pandas as pd
import argparse
import os
import json
from copy import deepcopy
from scipy import stats

import torch

from TORCH_OBJECTS import *
from source.td_opswtw import GROUP_ENVIRONMENT
import source.MODEL__Actor.grouped_actors2 as A_Module
import op_utils.instance as u_i

NEG_INF = Tensor([-np.inf])
P_VALUE_THRESHOLD = 0.05


def get_expected_tour(cur_env, actor):
    env = GROUP_ENVIRONMENT(group_env=cur_env, batch_s=1, group_s=1, use_expected=True)

    done = False
    total_reward = 0
    while not done:
        action_probs = actor.get_action_probabilities(env.group_state)
        # shape = (batch, group, problem)

        action = action_probs.argmax(dim=2)
        # shape = (batch, group)
        action[env.group_state.finished] = 0  # stay at depot, if you are finished
        _, reward, done = env.step(action + 1)
        total_reward += reward[0,0]

    return total_reward, env.group_state.selected_node_list[0, 0]


def evaluate_expected_tour(cur_env, expected_tour, args, state=None):
    batch_s = args.mc_num_samples_per_move
    group_s = 1

    env = GROUP_ENVIRONMENT(group_env=cur_env, batch_s=batch_s, group_s=group_s, deterministic=True)

    if state is not None:
        env.group_state.adj = state.adj

    cur_move = cur_env.group_state.selected_count

    for move in expected_tour[cur_move:]:
        env.step(move.repeat(batch_s, group_s))

    return env.group_state.rewards


def mc_simulation(cur_env, actor, args, return_state=False):
    n_samples = args.mc_num_samples_per_move
    n_moves = args.mc_num_moves
    n_moves_2 = args.mc_second_step_num
    batch_s = n_samples

    if cur_env.group_state.selected_count == 1 and args.mc_all_first:
        n_moves = (~cur_env.group_state.ninf_mask.eq(NEG_INF)).sum()
    elif cur_env.group_state.selected_count <= 5:
        n_moves *= 3

    group_s = n_moves * n_moves_2
    env = GROUP_ENVIRONMENT(group_env=cur_env, batch_s=batch_s, group_s=group_s, deterministic=True)


    group_reward = Tensor(np.zeros((batch_s, group_s)))

    if cur_env.group_state.selected_count == 1 and args.mc_all_first:
        first_actions = torch.masked_select(torch.arange(1, cur_env.group_state.problem_s+1, device=device), ~cur_env.group_state.ninf_mask.eq(NEG_INF))
        first_actions = first_actions.repeat(batch_s, 1)
        first_actions = first_actions.repeat_interleave(n_moves_2, dim=1)
    else:
        action_probs = actor.get_action_probabilities(env.group_state)
        # shape = (n_samples, n_moves, problem)

        # Use only the first index (because all tours in group are in same state at depot)
        prob_values, first_actions = action_probs[:, 0, :].topk(k=n_moves, dim=1)

        first_actions[prob_values == 0] = 0

        first_actions = first_actions.repeat_interleave(n_moves_2, dim=1)+1

    group_state, reward, done = env.step(first_actions)
    group_reward += reward

    diff_tour_indices = [i for i in range(0, group_s, n_moves_2)]

    if n_moves_2 > 1:
        action_probs = actor.get_action_probabilities(env.group_state)
        prob_values, second_actions = action_probs[:, diff_tour_indices, :].topk(k=n_moves_2, dim=2)
        second_actions = second_actions.reshape(batch_s, group_s)
        second_actions[group_state.finished] = 0

        batch_idx_mat = torch.arange(batch_s)[:, None].expand(batch_s, group_s)
        group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
        chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, second_actions].reshape(batch_s, group_s)
        # shape = (batch, group)
        second_actions[chosen_action_prob == 0]  = 0
        group_state, reward, done = env.step(second_actions+1)
        group_reward += reward

    while not done:
        action_probs = actor.get_action_probabilities(group_state)
        # shape = (batch, group, problem)

        if args.mc_sample:
            action = action_probs.reshape(batch_s * group_s, -1).multinomial(1)\
                .squeeze(dim=1).reshape(batch_s, group_s)
        else:
            action = action_probs.argmax(dim=2)
        # shape = (batch, group)
        action[group_state.finished] = 0  # stay at depot, if you are finished
        group_state, reward, done = env.step(action + 1)
        group_reward += reward

    if args.mc_combine_first and n_moves_2 > 1:
        avg_weights = torch.zeros(n_moves, group_s, device=device)
        same_first_indices = torch.repeat_interleave(torch.arange(n_moves), n_moves_2)
        avg_weights[same_first_indices, torch.arange(group_s)] = 1 / n_moves_2
        avg_per_first = torch.mm(avg_weights, group_reward.mean(dim=0)[:, None])
        return first_actions[0][diff_tour_indices], avg_per_first

    if return_state:
        return first_actions[0], group_reward, group_state

    return first_actions[0], group_reward


def get_next_move(cur_env, group_actor, mc_actor, args):
    if args.mc:
        actions, expected_rwds, state = mc_simulation(cur_env, mc_actor, args, return_state=True)
        action_index = expected_rwds.mean(dim=0).argmax(dim=0)
        action = LongTensor([[actions[action_index]]])
        if args.compare_baseline:
            baseline_expected_rwds, baseline_tour = get_expected_tour(cur_env, group_actor)
            baseline_rollout_results = evaluate_expected_tour(cur_env, baseline_tour, args, state=state)
            result = stats.ttest_rel(expected_rwds[:, action_index].cpu(), baseline_rollout_results[:, 0].cpu(), alternative='less')
            if result.pvalue < P_VALUE_THRESHOLD:
                # print(f'Reverting to Baseline Tour with P-value {result.pvalue}')
                action = LongTensor([[baseline_tour[cur_env.group_state.selected_count]]])
    else:
        action_probs = group_actor.get_action_probabilities(cur_env.group_state)
        action = action_probs.argmax(dim=2) + 1
    action[cur_env.group_state.finished] = 1

    return action


def create_tours(args):
    problem_size_list = [20, 50, 100, 200]
    problem_size = problem_size_list[(args.index - 1) // 250]
    inst_name = f'instance{args.index:04}'
    rwd_save_path = os.path.join(args.tour_dir, 'instance_rewards.csv')
    all_rwds_save_path = os.path.join(args.tour_dir, 'all_tour_rewards.csv')

    hyper_params_file = os.path.join(args.base_load_dir, str(problem_size), 'used_HYPER_PARAMS.txt')
    hp_dict = dict()
    with open(hyper_params_file, 'r') as f:
        lines = f.read().strip().split('\n')
        for hp in lines[2:]:
            k,v = hp.split(' = ')
            hp_dict[k] = v

    grouped_actor = A_Module.ACTOR(embedding_dim=int(hp_dict['EMBEDDING_DIM']), head_num=int(hp_dict['HEAD_NUM']),
                                   logit_clipping=int(hp_dict['LOGIT_CLIPPING']),
                                   encoder_layer_num=int(hp_dict['ENCODER_LAYER_NUM']),
                                   ff_hidden_dim=int(hp_dict['FF_HIDDEN_DIM']),
                                   key_dim=int(hp_dict['KEY_DIM'])).to(device)

    mc_actor = A_Module.ACTOR(embedding_dim=int(hp_dict['EMBEDDING_DIM']), head_num=int(hp_dict['HEAD_NUM']),
                              logit_clipping=int(hp_dict['LOGIT_CLIPPING']),
                              encoder_layer_num=int(hp_dict['ENCODER_LAYER_NUM']),
                              ff_hidden_dim=int(hp_dict['FF_HIDDEN_DIM']),
                              key_dim=int(hp_dict['KEY_DIM'])).to(device)

    actor_model_save_path = os.path.join(args.base_load_dir, str(problem_size), 'ACTOR_state_dic.pt')

    grouped_actor.load_state_dict(torch.load(actor_model_save_path, map_location=device))
    grouped_actor.eval()

    mc_actor.load_state_dict(torch.load(actor_model_save_path, map_location=device))
    mc_actor.eval()

    json_out_dict = dict()

    tour_rwds = []
    tour_lengths = []

    # Use separate random number generator for calculating travel times
    # Note that the MC Simulation does not have access to this number generator
    rng = np.random.RandomState(args.seed)
    batch_s= 1
    with torch.no_grad():
        print(f'Create Tours for {inst_name}, using seed {args.seed}')
        x, adj, _ = u_i.read_instance(os.path.join(args.data_dir, 'instances', f'{inst_name}.csv'),
                                      os.path.join(args.data_dir, 'adj-instances', f'adj-{inst_name}.csv'))
        group_s = 1
        inst_dict = {"nodes": problem_size, "seed": args.seed, "tours": dict()}

        env = GROUP_ENVIRONMENT(Tensor(x[np.newaxis, :, :]), Tensor(adj[np.newaxis, :, :]), deterministic=bool(args.deterministic), use_expected=bool(args.use_expected))
        group_state, _, _ = env.reset(group_s)
        mc_actor.reset(group_state)
        grouped_actor.reset(group_state)

        encoding_save_path = os.path.join(args.encoding_load_dir, f'{inst_name}_encoding.pt')
        if os.path.isfile(encoding_save_path):
            print('Loading saved encoding')
            encoding_dict = torch.load(encoding_save_path, map_location=device)

            # Set Values
            grouped_actor.encoded_graph = encoding_dict['Graph']
            grouped_actor.encoded_nodes = encoding_dict['Nodes']
            grouped_actor.node_prob_calculator.reset(grouped_actor.encoded_nodes)
            grouped_actor.node_prob_calculator.single_head_key = encoding_dict['Encoding']

            mc_actor.encoded_graph = encoding_dict['Graph'].clone()
            mc_actor.encoded_nodes = encoding_dict['Nodes'].clone()
            mc_actor.node_prob_calculator.reset(mc_actor.encoded_nodes)
            mc_actor.node_prob_calculator.single_head_key = encoding_dict['Encoding'].clone()

        mc_actor.increase_batch_size(args.mc_num_samples_per_move)
        for tour_index in range(1, 101):
            group_state, reward, done = env.reset(group_s)
            total_reward = Tensor(np.zeros((batch_s, group_s)))

            while not done:
                # if args.mc:
                #     actions, expected_rwds = mc_simulation(env, mc_actor, args)
                #     action_index = expected_rwds.argmax(dim=0)
                #     action = LongTensor([[actions[action_index]]])
                # else:
                #     action_probs = grouped_actor.get_action_probabilities(group_state)
                #     action = action_probs.argmax(dim=2) + 1
                # action[group_state.finished] = 1
                action = get_next_move(env, grouped_actor, mc_actor, args)
                group_state, reward, done = env.step(action, rng=rng)
                total_reward += reward

            tour_rwds.append(float(total_reward[0, 0].cpu()))
            tour = list(group_state.selected_node_list[0,0].cpu().numpy())
            tour_lengths.append(len(tour))
            other_nodes = set(range(1, problem_size+1)).difference(set(tour))
            for _ in range(len(other_nodes)):
                _ = rng.randint(1, 101, size=1)
            tour += list(other_nodes)
            inst_dict['tours'][f'tour{tour_index:03}'] = [int(v) for v in tour]

        json_out_dict[inst_name] = inst_dict

    json_write_path = os.path.join(args.tour_dir, f'{inst_name}.json')
    with open(json_write_path, 'w') as f:
        json.dump(json_out_dict, f)

    print(np.mean(tour_rwds))

    df = pd.DataFrame(columns=['Instance Index', 'Instance Name', 'Average Reward', 'Min Reward', 'Max Reward'])
    df = df.append({'Instance Index': args.index,
                    'Instance Name': inst_name,
                    'Average Reward': np.mean(tour_rwds),
                    'Min Reward': np.min(tour_rwds),
                    'Max Reward': np.max(tour_rwds)},
                   ignore_index=True)
    df.to_csv(rwd_save_path, index=False, mode='a', header=not os.path.isfile(rwd_save_path))

    if args.store_all_rewards:
        df_all_rwds = pd.DataFrame(data={'Instance Index': [args.index] * 100, 'Tour Index': [i for i in range(1, 101)],
                                         'Tour Reward': tour_rwds, 'Tour Length': tour_lengths})
        df_all_rwds.to_csv(all_rwds_save_path, index=False, mode='a', header=not os.path.isfile(all_rwds_save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Trained Model')
    parser.add_argument('index', type=int, help='Index of instance to run active search on (uses 1-indexing)')
    parser.add_argument('--tour_dir', default='./tours', help='Dir where tours are saved to. Defaults to "./tours"')
    parser.add_argument('--base_load_dir', default='./models/base',
                        help='Dir where base models are stored. Defaults to "./models/base/"')
    parser.add_argument('--seed', type=int, default=19120623, help='Random seed to use when generating tours.')
    parser.add_argument('--mc_num_moves', type=int, default=5, help='Number of moves to run monte carlo rollouts on.')
    parser.add_argument('--mc_num_samples_per_move', type=int, default=600, help='Number of samples to rollout for each '
                                                                               'of the top moves.')
    parser.add_argument('--mc_second_step_num', type=int, default=1, help='Number of second actions to take for each first in MC')
    parser.add_argument('--mc_all_first', action='store_true', help="Go to all feasible first nodes")
    parser.add_argument('--mc', action='store_true', help='Turn on Monte Carlo Simulations')
    parser.add_argument('--encoding_load_dir', default='./models/encodings',
                        help='Root directory to store best encodings. Defaults to "./models/encodings"')
    parser.add_argument('--mc_combine_first', action='store_true', help='Combine the expected value of all second moves')
    parser.add_argument('--data_dir', default='./data/test', help='Dir where data is stored. Defaults to "./data/test"')
    parser.add_argument('--mc_sample', action='store_true', help='Monte Carlo samples from distribution instead of greedy')
    parser.add_argument('--use_expected', action='store_true', help='Use the expected travel times for each step')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic model envs')
    parser.add_argument('--store_all_rewards', action='store_true', help='Saves the rewards for each tour')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare each action to the expected env greedy result')
    run_args = parser.parse_args()

    create_tours(run_args)


