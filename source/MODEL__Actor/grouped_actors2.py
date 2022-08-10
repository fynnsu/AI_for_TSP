
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

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# For debugging

from TORCH_OBJECTS import *


########################################
# ACTOR
########################################

class ACTOR(nn.Module):

    def __init__(self, embedding_dim, encoder_layer_num, head_num, key_dim, logit_clipping, ff_hidden_dim):
        super().__init__()

        self.encoder = Encoder(embedding_dim, encoder_layer_num, head_num, key_dim, ff_hidden_dim)
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_group(embedding_dim, head_num, key_dim, logit_clipping)

        self.batch_s = None
        self.encoded_nodes = None
        self.encoded_graph = None

    def reset(self, group_state):
        self.batch_s = group_state.data.size(0)
        self.encoded_nodes = self.encoder(group_state.data, group_state.adj)
        # shape = (batch, problem, EMBEDDING_DIM)
        self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
        # shape = (batch, 1, EMBEDDING_DIM)

        self.node_prob_calculator.reset(self.encoded_nodes)

    def load(self, encoded_nodes, encoded_graph, encoding):
        self.batch_s = encoded_nodes.size(0)
        self.encoded_nodes = encoded_nodes
        self.encoded_graph = encoded_graph

        self.node_prob_calculator.reset(self.encoded_nodes)

        self.node_prob_calculator.single_head_key = encoding.clone()

    def increase_batch_size(self, batch_s):
        assert self.batch_s == 1, "Starting batch size must be 1"
        self.batch_s = batch_s
        self.encoded_nodes = self.encoded_nodes.repeat(batch_s, 1, 1)
        self.encoded_graph = self.encoded_graph.repeat(batch_s, 1, 1)

        self.node_prob_calculator.k = self.node_prob_calculator.k.repeat(batch_s, 1, 1, 1)
        self.node_prob_calculator.v = self.node_prob_calculator.v.repeat(batch_s, 1, 1, 1)
        self.node_prob_calculator.single_head_key = self.node_prob_calculator.single_head_key.repeat(batch_s, 1, 1)

    def get_action_probabilities(self, group_state):
        encoded_LAST_NODES = pick_nodes_for_each_group(self.encoded_nodes, group_state.current_nodes-1)
        # shape = (batch, group, EMBEDDING_DIM)
        current_time = (group_state.tour_time / group_state.max_times).unsqueeze(dim=2)
        # shape = (batch, group, 1)

        item_select_probabilities = self.node_prob_calculator(self.encoded_graph, encoded_LAST_NODES,
                                                              current_time, ninf_mask=group_state.ninf_mask)
        # shape = (batch, group, problem+1)

        return item_select_probabilities


########################################
# ACTOR_SUB_NN : ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self, embedding_dim, encoder_layer_num, head_num, key_dim, ff_hidden_dim):
        super().__init__()
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(5, embedding_dim)
        self.layers = nn.ModuleList([Encoder_Layer(embedding_dim, head_num, key_dim, ff_hidden_dim) for _ in range(encoder_layer_num)])

    def forward(self, data, adj):
        # data.shape = (batch, problem, 7)

        depot_x = data[:, 0, 1] / 200 # scale xy to [0,1]
        # shape = (batch)
        depot_y = data[:, 0, 2] / 50
        # shape = (batch)

        node_x = data[:, 1:, [1]] / 200
        # shape = (batch, problem-1, 1)
        node_y = data[:, 1:, [2]] / 50
        # shape = (batch, problem-1, 1)

        node_tw = data[:, 1:, 3:5]
        # shape = (batch, problem-1, 2)
        scaled_node_tw = node_tw / (data[:, 0, 6][:, None, None])
        # shape = (batch, problem-1, 2)
        node_prize = data[:, 1:, [5]]
        # shape = (batch, problem-1, 1)

        node_xy_tw_prize = torch.cat((node_x, node_y, scaled_node_tw, node_prize), dim=2)
        # shape = (batch, problem-1, 5)

        embedded_depot = self.embedding_depot(torch.cat((depot_x[:, None, None], depot_y[:, None, None]), dim=2))
        # shape = (batch, 1, EMBEDDING_DIM)
        embedded_node = self.embedding_node(node_xy_tw_prize)
        # shape = (batch, problem-1, EMBEDDING_DIM)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape = (batch, problem, EMBEDDING_DIM)

        for layer in self.layers:
            out = layer(out)

        return out


class Encoder_Layer(nn.Module):
    def __init__(self, embedding_dim, head_num, key_dim, ff_hidden_dim):
        super().__init__()

        self.Wq = nn.Linear(embedding_dim, head_num * key_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * key_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * key_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * key_dim, embedding_dim)
        self.head_num = head_num

        self.addAndNormalization1 = Add_And_Normalization_Module(embedding_dim)
        self.feedForward = Feed_Forward_Module(embedding_dim, ff_hidden_dim)
        self.addAndNormalization2 = Add_And_Normalization_Module(embedding_dim)

    def forward(self, input1):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        q = reshape_by_heads(self.Wq(input1), head_num=self.head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=self.head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=self.head_num)
        # q shape = (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)
        # shape = (batch, problem, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


########################################
# ACTOR_SUB_NN : Next_Node_Probability_Calculator
########################################

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

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem+1, EMBEDDING_DIM)

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape = (batch, HEAD_NUM, problem+1, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch, EMBEDDING_DIM, problem+1)

    def forward(self, input1, input2, current_time, ninf_mask=None):
        # input1.shape = (batch, 1, EMBEDDING_DIM)
        # input2.shape = (batch, group, EMBEDDING_DIM)
        # remaining_loaded.shape = (batch, group, 1)
        # ninf_mask.shape = (batch, group, problem+1)

        group_s = input2.size(1)

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((input1.expand(-1, group_s, -1), input2, current_time), dim=2)
        # shape = (batch, group, 2*EMBEDDING_DIM+1)

        q = reshape_by_heads(self.Wq(input_cat), head_num=self.head_num)
        # shape = (batch, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=ninf_mask)
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


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def pick_nodes_for_each_group(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape = (batch, problem, EMBEDDING_DIM)
    # node_index_to_pick.shape = (batch, group)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(-1, -1, embedding_dim)
    # shape = (batch, group, EMBEDDING_DIM)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape = (batch, group, EMBEDDING_DIM)

    return picked_nodes


def reshape_by_heads(qkv, head_num):
    # q.shape = (batch, C, head_num*key_dim)

    batch_s = qkv.size(0)
    C = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, C, head_num, -1)
    # shape = (batch, C, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, problem)

    score_scaled = score / np.sqrt(key_dim)
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim
        self.embedding_dim = embedding_dim

    def forward(self, input1, input2):
        # input.shape = (batch, problem, EMBEDDING_DIM)
        batch_s = input1.size(0)
        problem_s = input1.size(1)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, self.embedding_dim))

        return normalized.reshape(batch_s, problem_s, self.embedding_dim)


class Feed_Forward_Module(nn.Module):
    def __init__(self, embedding_dim, ff_hidden_dim):
        super().__init__()

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        return self.W2(F.relu(self.W1(input1)))
