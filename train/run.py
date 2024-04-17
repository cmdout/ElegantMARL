import os
import sys
import time
import torch
import numpy as np
import torch.multiprocessing as mp  # torch.multiprocessing extends multiprocessing of Python
from copy import deepcopy
from multiprocessing import Process, Pipe

from buffer.replay_buffer import HAReplayBuffer
from HASAC_Train.config import Config, build_env
import pdb

from HASAC_Train.evaluator import Evaluator, get_cumulative_rewards_and_steps

if os.name == 'nt':  # if is WindowOS (Windows NT)
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
'''train'''


def train_agent_multiprocessing(args: Config):
    args.init_before_training()

    """Don't set method='fork' when send tensor in GPU"""
    method = 'spawn' if os.name == 'nt' else 'forkserver'  # os.name == 'nt' means Windows NT operating system (WinOS)
    mp.set_start_method(method=method, force=True)

    '''build the Pipe'''
    worker_pipes = [Pipe(duplex=False) for _ in range(args.num_workers)]  # receive, send
    learner_pipe = Pipe(duplex=False)
    evaluator_pipe = Pipe(duplex=True)

    '''build Process'''
    learner = Learner(learner_pipe=learner_pipe, worker_pipes=worker_pipes, evaluator_pipe=evaluator_pipe, args=args)
    workers = [Worker(worker_pipe=worker_pipe, learner_pipe=learner_pipe, worker_id=worker_id, args=args)
               for worker_id, worker_pipe in enumerate(worker_pipes)]
    evaluator = EvaluatorProc(evaluator_pipe=evaluator_pipe, args=args)

    '''start Process'''
    process_list = [learner, *workers, evaluator]
    [process.start() for process in process_list]
    [process.join() for process in process_list]


class Learner(Process):
    def __init__(self, learner_pipe: Pipe, worker_pipes: [Pipe], evaluator_pipe: Pipe, args: Config):
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [worker_pipe[1] for worker_pipe in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.args = args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)
        # pdb.set_trace()
        '''init buffer'''
        buffer = [HAReplayBuffer(args.buffer_size,
                                 args.state_dim[i],
                                 args.obs_dim[i],
                                 args.action_dim[i],
                                 args.gpu_id,
                                 args.num_envs * args.num_workers,
                                 args) for i in range(args.agent_num)]
        # pdb.set_trace()

        '''loop'''
        if_off_policy = args.if_off_policy
        if_save_buffer = args.if_save_buffer
        num_workers = args.num_workers
        num_envs = args.num_envs
        state_dim = args.state_dim
        obs_dim = args.obs_dim
        action_dim = args.action_dim
        horizon_len = args.horizon_len
        num_seqs = args.num_envs * args.num_workers
        num_steps = args.horizon_len * args.num_workers
        agent_num = args.agent_num
        cwd = args.cwd
        del args

        # agent.last_state = torch.empty((num_seqs, state_dim), dtype=torch.float32, device=agent.device)
        next_stats = [torch.empty((horizon_len, num_seqs, state_dim[i]),
                                        dtype=torch.float32,
                                        device=agent.device
                                        ) for i in range(agent_num)]

        states = [torch.empty((horizon_len, num_seqs, state_dim[i]),
                              dtype=torch.float32,
                              device=agent.device
                              )for i in range(agent_num)]
        obs = [torch.empty((horizon_len, num_seqs, obs_dim[i]),
                              dtype=torch.float32,
                              device=agent.device
                              )for i in range(agent_num)]
        actions = [torch.empty((horizon_len, num_seqs, action_dim[i]),
                               dtype=torch.float32,
                               device=agent.device
                               )for i in range(agent_num)]
        rewards = [torch.empty((horizon_len, num_seqs),
                               dtype=torch.float32,
                               device=agent.device
                               )for i in range(agent_num)]
        dones = [torch.empty((horizon_len, num_seqs),
                               dtype=torch.bool,
                               device=agent.device
                               )for i in range(agent_num)]
        logprobs = [torch.empty((horizon_len, num_seqs),
                                dtype=torch.float32,
                                device=agent.device
                                )for i in range(agent_num)]
        buffer_items_tensor = (states, next_stats, obs, actions, logprobs, rewards, dones)

        if_train = True
        while if_train:
            '''Learner send actor to Workers'''
            for send_pipe in self.send_pipes:
                send_pipe.send(agent.agent)

            '''Learner receive (buffer_items, last_state) from Workers'''
            for _ in range(num_workers):
                worker_id, buffer_items = self.recv_pipe.recv()

                buf_i = worker_id * num_envs
                buf_j = worker_id * num_envs + num_envs
                for buffer_item, buffer_tensor in zip(buffer_items, buffer_items_tensor):
                    for i in range(agent.agent_num):
                        buffer_tensor[i][:, buf_i:buf_j] = buffer_item[i].unsqueeze(1)

            '''Learner update training data to (buffer, agent)'''
            for i in range(agent.agent_num):
                buffer[i].update((states[i], next_stats[i], obs[i], actions[i], logprobs[i], rewards[i], dones[i]))

            '''agent update network using training data'''
            torch.set_grad_enabled(True)
            #pdb.set_trace()
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            '''Learner receive training signal from Evaluator'''
            if self.eval_pipe.poll():  # whether there is any data available to be read of this pipe
                if_train = self.eval_pipe.recv()  # True means evaluator in idle moments.
                actor = agent.act  # so Leaner send an actor to evaluator for evaluation.
            else:
                actor = None

            '''Learner send actor and training log to Evaluator'''
            exp_r = buffer_items_tensor[2].mean().item()  # the average rewards of exploration
            self.eval_pipe.send((actor, num_steps, exp_r, logging_tuple))

        '''Learner send the terminal signal to workers after break the loop'''
        for send_pipe in self.send_pipes:
            send_pipe.send(None)

        '''save'''
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)
            print(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}")


class Worker(Process):
    def __init__(self, worker_pipe: Pipe, learner_pipe: Pipe, worker_id: int, args: Config):
        super().__init__()
        self.recv_pipe = worker_pipe[0]
        self.send_pipe = learner_pipe[1]
        self.worker_id = worker_id
        self.args = args

    def run(self):
        args = self.args
        worker_id = self.worker_id
        torch.set_grad_enabled(False)

        '''init environment'''
        env = build_env(args.env_class, args.env_args, args.gpu_id)
        # pdb.set_trace()

        '''init agent'''
        agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
        agent.save_or_load_agent(args.cwd, if_save=False)

        '''loop'''
        horizon_len = args.horizon_len
        del args

        while True:
            '''Worker receive actor from Learner'''
            actor = self.recv_pipe.recv()
            if actor is None:
                break

            '''Worker send the training data to Learner'''
            agent.agent = actor
            buffer_items = agent.explore_env(env, horizon_len)
            self.send_pipe.send((worker_id, buffer_items))

        env.close() if hasattr(env, 'close') else None


class EvaluatorProc(Process):
    def __init__(self, evaluator_pipe: Pipe, args: Config):
        super().__init__()
        self.pipe = evaluator_pipe[0]
        self.args = args

    def run(self):
        args = self.args
        torch.set_grad_enabled(False)

        '''wandb(weights & biases): Track and visualize all the pieces of your machine learning pipeline.'''
        wandb = None
        if getattr(args, 'if_use_wandb', False):
            import wandb
            wandb_project_name = "train"
            wandb.init(project=wandb_project_name)

        '''init evaluator'''
        eval_env_class = args.eval_env_class if args.eval_env_class else args.env_class
        eval_env_args = args.eval_env_args if args.eval_env_args else args.env_args
        eval_env = build_env(eval_env_class, eval_env_args, args.gpu_id)
        evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_tensorboard=False)

        '''loop'''
        cwd = args.cwd
        break_step = args.break_step
        device = torch.device(f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and (args.gpu_id >= 0)) else "cpu")
        del args

        if_train = True
        while if_train:
            '''Evaluator receive training log from Learner'''
            actor, steps, exp_r, logging_tuple = self.pipe.recv()
            wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]}) if wandb else None

            '''Evaluator evaluate the actor and save the training log'''
            if actor is None:
                evaluator.total_step += steps  # update total_step but don't update recorder
            else:
                actor = actor.to(device)
                evaluator.evaluate_and_save(actor, steps, exp_r, logging_tuple)

            '''Evaluator send the training signal to Learner'''
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe.send(if_train)

        '''Evaluator save the training log and draw the learning curve'''
        evaluator.save_training_curve_jpg()
        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        eval_env.close() if hasattr(eval_env, 'close') else None


'''render'''


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
    env = build_env(env_class, env_args)

    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act
    del agent

    print(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_cumulative_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")
