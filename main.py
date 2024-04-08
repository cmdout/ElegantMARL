from argparse import ArgumentParser

from agent import AgentHAPPO
from envs.MpeEnv import PettingZooMPEEnv
from train.config import get_mpe_env_args, Config
from train.run import train_agent_multiprocessing


def train_happo_for_mpe(args):
    agent_class = getattr(AgentHAPPO, 'AgentHAPPO', None)
    env_class = PettingZooMPEEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = get_mpe_env_args(args, env=PettingZooMPEEnv(continuous_actions=args['continuous_actions'], scenario=args['scenario'], num_envs=args['num_envs']), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(8e4)  # break training if 'total_step > break_step'
    args.net_dims = (128, 64)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step * 4

    args.repeat_times = 32  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.if_build_vec_env = True
    train_agent_multiprocessing(args)  # train_agent(args)


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=0, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=str, default='1', help='the environment ID for training')
    Parser.add_argument('--scenario', type=str, default='simple_spread_v2', help='the environment name')
    Parser.add_argument('--continuous_actions', type=bool, default=True, help='the action type')
    Parser.add_argument('--num_envs', type=int, default=1, help='the number of sub env')
    Parser.add_argument('--num_workers', type=int, default=1, help='the number of workers')
    Args = Parser.parse_args()
    args = vars(Args)
    train_happo_for_mpe(args)