#  MIT License
#
#  Copyright (c) 2017 Ilya Kostrikov and (c) 2020 Google LLC
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.a2c_ppo_acktr.distributions import PlainDiagGaussian
from third_party.a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class SplitPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(SplitPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        num_outputs = action_space.shape[0]

        self.base = SplitPolicyBase(obs_shape[0], num_outputs, **base_kwargs)
        self.dist = PlainDiagGaussian(num_outputs)

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        # not used
        raise NotImplementedError

    def reset_variance(self, action_space, log_std):
        num_outputs = action_space.shape[0]
        # TODO
        self.dist.reset_variance(num_outputs, log_std)

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_mean, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_mean)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value,  actor_mean, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_mean)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class SplitPolicyBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=64):

        # split actor into 4 nets here
        # last layer is linear and gain is 1.0 rather than 1.414
        # probably need to move last layer from distribution to here

        super(SplitPolicyBase, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        init_final_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        init_final_act_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                     constant_(x, 0), gain=0.02)

        assert (num_inputs // 4) * 4 == num_inputs      # multiple of 4

        self.each_num_input = num_inputs // 4
        self.each_num_output = num_outputs // 4

        self.actor_1 = nn.Sequential(
            init_(nn.Linear(self.each_num_input, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_final_act_(nn.Linear(hidden_size, self.each_num_output)))

        self.actor_2 = nn.Sequential(
            init_(nn.Linear(self.each_num_input, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_final_act_(nn.Linear(hidden_size, self.each_num_output)))

        self.actor_3 = nn.Sequential(
            init_(nn.Linear(self.each_num_input, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_final_act_(nn.Linear(hidden_size, self.each_num_output)))

        self.actor_4 = nn.Sequential(
            init_(nn.Linear(self.each_num_input, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_final_act_(nn.Linear(hidden_size, self.each_num_output)))

        # TODO: keep value function unsplit, since 4 actors share one reward from D
        self.critic_full = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_final_(nn.Linear(hidden_size, 1)))

        self.train()
        
    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        value = self.critic_full(x)

        # for _ in self.critic_full.parameters():
        #     print(_)
        #     break
        # print(self.actors[0].parameters()[0])

        # print(self.each_num_input)
        # print(x.size())
        # print(x[:self.each_num_input].size())
        
        y1 = self.actor_1(x[:, :self.each_num_input])
        y2 = self.actor_2(x[:, self.each_num_input:self.each_num_input*2])
        y3 = self.actor_3(x[:, self.each_num_input*2:self.each_num_input*3])
        y4 = self.actor_4(x[:, self.each_num_input*3:])

        action_mean = torch.cat((y1, y2, y3, y4), 1)

        return value, action_mean, rnn_hxs


