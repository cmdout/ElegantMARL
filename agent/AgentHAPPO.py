import torch
from copy import deepcopy
from typing import Tuple
from torch import Tensor
import numpy as np
from agent.AgentPPO import AgentPPO
from agent.net import CriticPPO
from envs.env_tool import check
from HASAC_Train.config import Config


class AgentHAPPO():
    def __init__(self, net_dims: [int], state_dim: list, action_dim: list, gpu_id: int = 0, args: Config = Config()):
        self.gamma = args.gamma  # discount factor of future rewards
        self.num_envs = args.num_envs  # the number of sub envs in vectorized env. `num_envs=1` in single env.
        self.batch_size = args.batch_size  # num of transitions sampled from replay buffer.
        self.repeat_times = args.repeat_times  # repeatedly update network using ReplayBuffer
        self.reward_scale = args.reward_scale  # an approximate target reward usually be closed to 256
        self.learning_rate = args.learning_rate  # the learning rate for network updating
        self.if_off_policy = args.if_off_policy  # whether off-policy or on-policy of DRL algorithm
        self.clip_grad_norm = args.clip_grad_norm  # clip the gradient after normalization
        self.soft_update_tau = args.soft_update_tau  # the tau of soft target update `net = (1-tau)*net + net1`
        self.state_value_tau = args.state_value_tau  # the tau of normalize for value and state
        self.agent_num = args.agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.obs_dim = args.obs_dim
        self.last_state = None  # last state of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.last_obs = None  # last obs of the trajectory for training. last_state.shape == (num_envs, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.agent = [AgentPPO(net_dims, self.obs_dim[i], action_dim[i]) for i in range(self.agent_num)]
        self.cri = CriticPPO(net_dims, state_dim[0], action_dim[0]).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.cri.parameters(), lr=1e-4)
        self.critic_target = deepcopy(self.cri)
        self.action_spaces = args.action_spaces

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        env: RL training environment. env.reset() env.step(). It should be a vector env.
        horizon_len: collect horizon_len step while exploring to update networks
        return: `(states, actions, rewards, undones)` for off-policy
            env_num == 1
            states.shape == (horizon_len, env_num, state_dim)
            actions.shape == (horizon_len, env_num, action_dim)
            logprobs.shape == (horizon_len, env_num, action_dim)
            rewards.shape == (horizon_len, env_num)
            undones.shape == (horizon_len, env_num)
        """
        states = [torch.zeros((horizon_len, self.state_dim[i]),
                              dtype=torch.float32,
                              device=self.device
                              ) for i in range(self.agent_num)]
        next_states = [torch.zeros((horizon_len, self.state_dim[i]),
                              dtype=torch.float32,
                              device=self.device
                              ) for i in range(self.agent_num)]
        obs_ = [torch.zeros((horizon_len, self.obs_dim[i]),
                            dtype=torch.float32,
                            device=self.device
                            ) for i in range(self.agent_num)]
        actions = [torch.zeros((horizon_len, self.action_dim[i]),
                               dtype=torch.float32,
                               device=self.device
                               ) for i in range(self.agent_num)]
        rewards = [torch.zeros(horizon_len,
                               dtype=torch.float32,
                               device=self.device
                               ) for i in range(self.agent_num)]
        dones = [torch.empty(horizon_len,
                             dtype=torch.bool,
                             device=self.device
                             ) for i in range(self.agent_num)]
        logprobs = [torch.zeros(horizon_len,
                                dtype=torch.float32,
                                device=self.device
                                ) for i in range(self.agent_num)]

        obs, state, _ = env.reset()
        get_action = self.get_action
        for t in range(horizon_len):
            action, logprob = get_action(obs)
            for i in range(self.agent_num):
                states[i][t] = torch.as_tensor(state[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                obs_[i][t] = torch.as_tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            if t == 24:
                print(1)
            next_obs, next_state, reward, done, _, _ = env.step(action)  # next_state
            if np.all(done):
                next_obs, next_state, _ = env.reset() 
            next_state = [torch.as_tensor(next_state[i], dtype=torch.float32, device=self.device).unsqueeze(0)for i in range(self.agent_num)]
            next_obs = [torch.as_tensor(next_obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)for i in range(self.agent_num)]
            for i in range(self.agent_num):
                next_states[i][t] = torch.as_tensor(next_state[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                actions[i][t] = action[i]
                logprobs[i][t] = logprob[i]
                rewards[i][t] = reward[i][0]
                dones[i][t] = done[i]
            state = next_state
            obs = next_obs


        rewards *= self.reward_scale
        return states, next_states, obs_, actions, logprobs, rewards, dones

    def update(self, replay_buffer):
        print(1)
        return 1


    def get_action(self, obs_n):
        actions = []
        log_probs = []
        for i, obs in enumerate(obs_n):
            obs_input = check(obs, self.device)
            action, log_prob = self.agent[i].act.get_action(obs_input)
            action = self.agent[i].act.convert_action_for_env(action, self.action_spaces[i].high[0])
            actions.append(action)
            log_probs.append(log_prob)
        return actions, log_probs

    def prepare_buffer(self, batch):
        o = batch['o']
        a = batch['u']
        s = batch['s']
        r = batch['r']
        o_ = batch['o_next']
        s_ = batch['s_next']
        d = batch['terminated']
        log_probs = batch['log_probs']
        pre_s = batch['pre_state']
        all_done = batch['all_done']
        value = self.critic_target(torch.tensor(s, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
        prev_r_sum = self.critic(torch.tensor(pre_s, dtype=torch.float32).to(self.device)).detach().cpu().numpy()
        prev_value = copy.deepcopy(prev_r_sum)
        prev_advantage = np.zeros_like(prev_value)
        buf_r_sum = np.empty_like(r)
        buf_advantage = np.empty_like(r)
        for i in range(self.eposide_limit_step - 1, -1, -1):
            buf_r_sum[:, i] = np.squeeze(r[:, i].reshape(r.shape[0], 1) + d[:, i].reshape(d.shape[0], 1) * prev_r_sum)
            prev_r_sum = buf_r_sum[:, i].reshape(buf_r_sum.shape[0], 1)
            buf_advantage[:, i] = np.squeeze(
                r[:, i].reshape(r.shape[0], 1) + d[:, 1].reshape(d.shape[0], 1) * prev_value - value[:, i].reshape(
                    value.shape[0], 1) + d[:, i].reshape(d.shape[0], 1) * self.lambda_gae_adv * prev_advantage)
            prev_value = value[:, i].reshape(value.shape[0], 1)
            prev_advantage = buf_advantage[:, i].reshape(buf_advantage.shape[0], 1)
        # buf_advantage = (buf_advantage - np.mean(buf_advantage)) / (np.std(buf_advantage) + 1e-8)
        # 生成agent的随机排序
        arr = np.arange(self.agent_num)
        np.random.shuffle(arr)
        for num in arr:
            new_prob, new_entropy = self.actors[num].get_logprob_entropy(torch.tensor(o[:, :, num]).to(self.device),
                                                                         torch.tensor(a[:, :, num]).to(self.device))
            ratio = torch.exp(new_prob - torch.tensor(log_probs[:, :, num]).squeeze().to(self.device)).squeeze()
            surr1 = ratio * torch.tensor(buf_advantage, dtype=torch.float32).to(self.device)
            surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * torch.tensor(buf_advantage,
                                                                                        dtype=torch.float32).to(
                self.device)
            actor_loss = (-torch.min(surr1, surr2) + self.lambda_entropy * new_entropy.squeeze()).to(self.device)
            actor_loss = actor_loss * torch.tensor(all_done[:, :, num], dtype=torch.float32).squeeze().to(self.device)
            actor_loss = actor_loss.mean()
            self.actors_optimizer[num].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm(self.actors[num].parameters(), self.max_grad_norm)
            self.actors_optimizer[num].step()
            update_after_prob, update_after_entropy = self.actors[num].get_logprob_entropy(
                torch.tensor(o[:, :, num]).to(self.device), torch.tensor(a[:, :, num]).to(self.device))
            ratio = torch.exp(update_after_prob - new_prob).to(self.device).squeeze()
            buf_advantage = ratio * torch.tensor(buf_advantage, dtype=torch.float32).to(self.device)
        value = self.critic(torch.tensor(s).to(self.device)).to(self.device).squeeze(-1)
        value_loss = F.mse_loss(torch.tensor(buf_r_sum).to(self.device), value).to(self.device)
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        for parmas, target_par in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_par.data.copy_(self.tau * parmas.data + (1 - self.tau) * target_par.data)

    def save_or_load_agent(self, cwd: str, if_save: bool):
        for i in range(self.agent_num):
            self.agent[i].save_or_load_agent(cwd, if_save)
