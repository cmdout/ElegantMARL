import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import numpy as np
from envs.env_tool import check
from envs.env_tool import check1
from hasac_models.value_function_models.continuous_q_net import ContinuousQNet
import itertools
from copy import deepcopy
from utils.models_tools import update_linear_schedule
import torch.nn.functional as F
"""Actor (policy network)"""


class ActorBase(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim, *dims, action_dim])
        self.explore_noise_std = None  # standard deviation of exploration action noise
        self.ActionDist = torch.distributions.normal.Normal

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std


class ActorPPO(ActorBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, action_dim], activation=nn.Tanh)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        state = self.state_norm(state)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = self.ActionDist(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor, max_action) -> Tensor:
        return action*max_action


"""Critic (value network)"""


class CriticBase(nn.Module):  # todo state_norm, value_norm
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = None  # build_mlp(dims=[state_dim + action_dim, *dims, 1])

        self.state_avg = nn.Parameter(torch.zeros((state_dim,)), requires_grad=False)
        self.state_std = nn.Parameter(torch.ones((state_dim,)), requires_grad=False)
        self.value_avg = nn.Parameter(torch.zeros((1,)), requires_grad=False)
        self.value_std = nn.Parameter(torch.ones((1,)), requires_grad=False)

    def state_norm(self, state: Tensor) -> Tensor:
        return (state - self.state_avg) / self.state_std  # todo state_norm

    def value_re_norm(self, value: Tensor) -> Tensor:
        return value * self.value_std + self.value_avg  # todo value_norm


class Critic(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 1])

        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values.squeeze(dim=1)  # q value


class CriticTwin(CriticBase):  # shared parameter
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim + action_dim, *dims, 2])

        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values.mean(dim=1)  # mean Q value

    def get_q_min(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return torch.min(values, dim=1)[0]  # min Q value

    def get_q1_q2(self, state, action):
        state = self.state_norm(state)
        values = self.net(torch.cat((state, action), dim=1))
        values = self.value_re_norm(values)
        return values[:, 0], values[:, 1]  # two Q values


class CriticPPO(CriticBase):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.net = build_mlp(dims=[state_dim, *dims, 1])
        layer_init_with_orthogonal(self.net[-1], std=0.5)

    def forward(self, state: Tensor) -> Tensor:
        state = self.state_norm(state)
        value = self.net(state)
        value = self.value_re_norm(value)
        return value.squeeze(1)  # q value


"""utils"""


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)


class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        return torch.cat(
            (x2, self.dense2(x2)), dim=1
        )  # x3  # x2.shape==(-1, lay_dim*4)


class ConvNet(nn.Module):  # pixel-level state encoder
    def __init__(self, inp_dim, out_dim, image_size=224):
        super().__init__()
        if image_size == 224:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 224, 224)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=110
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(96, 128, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(128, 192, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(192, out_dim),  # size==(batch_size, out_dim)
            )
        elif image_size == 112:
            self.net = nn.Sequential(  # size==(batch_size, inp_dim, 112, 112)
                nn.Conv2d(inp_dim, 32, (5, 5), stride=(2, 2), bias=False),
                nn.ReLU(inplace=True),  # size=54
                nn.Conv2d(32, 48, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=26
                nn.Conv2d(48, 64, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=12
                nn.Conv2d(64, 96, (3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),  # size=5
                nn.Conv2d(96, 128, (5, 5), stride=(1, 1)),
                nn.ReLU(inplace=True),  # size=1
                NnReshape(-1),  # size (batch_size, 1024, 1, 1) ==> (batch_size, 1024)
                nn.Linear(128, out_dim),  # size==(batch_size, out_dim)
            )
        else:
            assert image_size in {224, 112}

    def forward(self, x):
        # assert x.shape == (batch_size, inp_dim, image_size, image_size)
        x = x.permute(0, 3, 1, 2)
        x = x / 128.0 - 1.0
        return self.net(x)

    @staticmethod
    def check():
        inp_dim = 3
        out_dim = 32
        batch_size = 2
        image_size = [224, 112][1]
        # from elegantrl.net import Conv2dNet
        net = ConvNet(inp_dim, out_dim, image_size)

        image = torch.ones((batch_size, image_size, image_size, inp_dim), dtype=torch.uint8) * 255
        print(image.shape)
        output = net(image)
        print(output.shape)


class TwinContinuousQCritic:
    """Twin Continuous Q Critic.
    Critic that learns two Q-functions. The action space is continuous.
    Note that the name TwinContinuousQCritic emphasizes its structure that takes observations and actions as input and
    outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be used in
    discrete action space. For now, it only supports continuous action space, but we will enhance its capability to
    include discrete action space in the future.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = act_space[0].__class__.__name__
        self.critic = ContinuousQNet(args, share_obs_space, act_space, device)
        self.critic2 = ContinuousQNet(args, share_obs_space, act_space, device)
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = args["use_proper_time_limits"]
        critic_params = itertools.chain(
            self.critic.parameters(), self.critic2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
        )
        self.turn_off_grad()

    def lr_decay(self, step, steps):
        """Decay the actor and critic learning rates.
        Args:
            step: (int) current training step.
            steps: (int) total number of training steps.
        """
        update_linear_schedule(self.critic_optimizer, step, steps, self.critic_lr)

    def soft_update(self):
        """Soft update the target networks."""
        for param_target, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )
        for param_target, param in zip(
            self.target_critic2.parameters(), self.critic2.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak) + param.data * self.polyak
            )

    def get_values(self, share_obs, actions):
        """Get the Q values for the given observations and actions."""
        share_obs = check1(share_obs).to(**self.tpdv)
        actions = check1(actions).to(**self.tpdv)
        return self.critic(share_obs, actions)

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """Train the critic.
        Args:
            share_obs: (np.ndarray) shape is (batch_size, dim)
            actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            reward: (np.ndarray) shape is (batch_size, 1)
            done: (np.ndarray) shape is (batch_size, 1)
            term: (np.ndarray) shape is (batch_size, 1)
            next_share_obs: (np.ndarray) shape is (batch_size, dim)
            next_actions: (np.ndarray) shape is (n_agents, batch_size, dim)
            gamma: (np.ndarray) shape is (batch_size, 1)
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check1(share_obs).to(**self.tpdv)
        actions = check1(actions).to(**self.tpdv)
        actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        reward = check1(reward).to(**self.tpdv)
        done = check1(done).to(**self.tpdv)
        term = check1(term).to(**self.tpdv)
        gamma = check1(gamma).to(**self.tpdv)
        next_share_obs = check1(next_share_obs).to(**self.tpdv)
        next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            q_targets = reward + gamma * next_q_values * (1 - term)
        else:
            q_targets = reward + gamma * next_q_values * (1 - done)
        critic_loss1 = torch.mean(
            torch.nn.functional.mse_loss(self.critic(share_obs, actions), q_targets)
        )
        critic_loss2 = torch.mean(
            torch.nn.functional.mse_loss(self.critic2(share_obs, actions), q_targets)
        )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, save_dir):
        """Save the model parameters."""
        torch.save(self.critic.state_dict(), str(save_dir) + "/critic_agent" + ".pt")
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent" + ".pt",
        )
        torch.save(self.critic2.state_dict(), str(save_dir) + "/critic_agent2" + ".pt")
        torch.save(
            self.target_critic2.state_dict(),
            str(save_dir) + "/target_critic_agent2" + ".pt",
        )

    def restore(self, model_dir):
        """Restore the model parameters."""
        critic_state_dict = torch.load(str(model_dir) + "/critic_agent" + ".pt")
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent" + ".pt"
        )
        self.target_critic.load_state_dict(target_critic_state_dict)
        critic_state_dict2 = torch.load(str(model_dir) + "/critic_agent2" + ".pt")
        self.critic2.load_state_dict(critic_state_dict2)
        target_critic_state_dict2 = torch.load(
            str(model_dir) + "/target_critic_agent2" + ".pt"
        )
        self.target_critic2.load_state_dict(target_critic_state_dict2)

    def turn_on_grad(self):
        """Turn on the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """Turn off the gradient for the critic network."""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
# hasac网络
class SoftTwinContinuousQCritic(TwinContinuousQCritic):
    """Soft Twin Continuous Q Critic.
    Critic that learns two soft Q-functions. The action space can be continuous and discrete.
    Note that the name SoftTwinContinuousQCritic emphasizes its structure that takes observations and actions as input
    and outputs the q values. Thus, it is commonly used to handle continuous action space; meanwhile, it can also be
    used in discrete action space.
    """

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """Initialize the critic."""
        super(SoftTwinContinuousQCritic, self).__init__(
            args, share_obs_space, act_space, num_agents, state_type, device
        )

        self.tpdv_a = dict(dtype=torch.int64, device=device)
        self.auto_alpha = args["auto_alpha"]
        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args["alpha_lr"]
            )
            self.alpha = torch.exp(self.log_alpha.detach())
        else:
            self.alpha = args["alpha"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

    def update_alpha(self, logp_actions, target_entropy):
        """Auto-tune the temperature parameter alpha."""
        log_prob = (
            torch.sum(torch.cat(logp_actions, dim=-1), dim=-1, keepdim=True)
            .detach()
            .to(**self.tpdv)
            + target_entropy
        )
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())

    def get_values(self, share_obs, actions):
        """Get the soft Q values for the given observations and actions."""
        share_obs = check1(share_obs).to(**self.tpdv)
        actions = check1(actions).to(**self.tpdv)
        return torch.min(
            self.critic(share_obs, actions), self.critic2(share_obs, actions)
        )

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        valid_transition,
        term,
        next_share_obs,
        next_actions,
        next_logp_actions,
        gamma,
        value_normalizer=None,
    ):
        """Train the critic.
        Args:
            share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            actions: (n_agents, batch_size, dim)
            reward: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            done: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            valid_transition: (n_agents, batch_size, 1)
            term: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            next_share_obs: EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            next_actions: (n_agents, batch_size, dim)
            next_logp_actions: (n_agents, batch_size, 1)
            gamma: EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        """
        assert share_obs.__class__.__name__ == "ndarray"
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert next_share_obs.__class__.__name__ == "ndarray"
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check1(share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions = check1(actions).to(**self.tpdv)
            actions = torch.cat([actions[i] for i in range(actions.shape[0])], dim=-1)
        else:
            actions = check1(actions).to(**self.tpdv_a)
            one_hot_actions = []
            for agent_id in range(len(actions)):
                if self.action_type == "MultiDiscrete":
                    action_dims = self.act_space[agent_id].nvec
                    one_hot_action = []
                    for dim in range(len(action_dims)):
                        one_hot = F.one_hot(
                            actions[agent_id, :, dim], num_classes=action_dims[dim]
                        )
                        one_hot_action.append(one_hot)
                    one_hot_action = torch.cat(one_hot_action, dim=-1)
                else:
                    one_hot_action = F.one_hot(
                        actions[agent_id], num_classes=self.act_space[agent_id].n
                    )
                one_hot_actions.append(one_hot_action)
            actions = torch.squeeze(torch.cat(one_hot_actions, dim=-1), dim=1).to(
                **self.tpdv_a
            )
        if self.state_type == "FP":
            actions = torch.tile(actions, (self.num_agents, 1))
        reward = check1(reward).to(**self.tpdv)
        done = check1(done).to(**self.tpdv)
        valid_transition = check1(np.concatenate(valid_transition, axis=0)).to(
            **self.tpdv
        )
        term = check1(term).to(**self.tpdv)
        gamma = check1(gamma).to(**self.tpdv)
        next_share_obs = check1(next_share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv)
        else:
            next_actions = torch.cat(next_actions, dim=-1).to(**self.tpdv_a)
        next_logp_actions = torch.sum(
            torch.cat(next_logp_actions, dim=-1), dim=-1, keepdim=True
        ).to(**self.tpdv)
        if self.state_type == "FP":
            next_actions = torch.tile(next_actions, (self.num_agents, 1))
            next_logp_actions = torch.tile(next_logp_actions, (self.num_agents, 1))
        next_q_values1 = self.target_critic(next_share_obs, next_actions)
        next_q_values2 = self.target_critic2(next_share_obs, next_actions)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.use_proper_time_limits:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check1(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - term)
                value_normalizer.update(q_targets)
                q_targets = check1(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values - self.alpha * next_logp_actions
                ) * (1 - term)
        else:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check1(value_normalizer.denormalize(next_q_values)).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - done)
                value_normalizer.update(q_targets)
                q_targets = check1(value_normalizer.normalize(q_targets)).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values - self.alpha * next_logp_actions
                ) * (1 - done)
        if self.use_huber_loss:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss1 = (
                    torch.sum(
                        F.huber_loss(
                            self.critic(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
                critic_loss2 = (
                    torch.mean(
                        F.huber_loss(
                            self.critic2(share_obs, actions),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss1 = torch.mean(
                    F.huber_loss(
                        self.critic(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
                critic_loss2 = torch.mean(
                    F.huber_loss(
                        self.critic2(share_obs, actions),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
        else:
            if self.state_type == "FP" and self.use_policy_active_masks:
                critic_loss1 = (
                    torch.sum(
                        F.mse_loss(self.critic(share_obs, actions), q_targets)
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
                critic_loss2 = (
                    torch.sum(
                        F.mse_loss(self.critic2(share_obs, actions), q_targets)
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss1 = torch.mean(
                    F.mse_loss(self.critic(share_obs, actions), q_targets)
                )
                critic_loss2 = torch.mean(
                    F.mse_loss(self.critic2(share_obs, actions), q_targets)
                )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()