from typing import List
import random
# import gym
import numpy as np
from parse import parse_state
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse
from env import *
from datetime import datetime
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='CartPole or LunarLander OpenAI gym environment', type=str)
parser.add_argument('--use_cuda', help='Use if you want to use CUDA', action='store_true')


class Params:
    NUM_EPOCHS = 1000
    ALPHA = 5e-3  # learning rate
    BATCH_SIZE = 1  # how many episodes we want to pack into an epoch
    GAMMA = 0.99  # discount rate
    HIDDEN_SIZE = 256  # number of hidden nodes we have in our dnn
    BETA = 0.1  # the entropy bonus multiplier
    STATE_SPACE = 3  # 1 row representation of SLURM squeue
    ACTION_SPACE = 2  # submit/no-submit task
    # state space : wait_time, time_limit, number of nodes


# Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.PReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = normalize(x, dim=1)
        x = self.net(x)
        return x


class RNN_Agent1(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(RNN_Agent1, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(observation_space_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(observation_space_size + hidden_size, action_space_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)


class RNN_Agent2(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers=1):
        super(RNN_Agent2, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        out, _ = self.rnn(input, h0)
        out = out[:, -1, :]
        out = self.fc(out)

        return out


class Baseline:
    def __init__(self, problem: str = "CartPole", use_cuda: bool = False):
        self.NUM_EPOCHS = Params.NUM_EPOCHS
        self.BATCH_SIZE = Params.BATCH_SIZE
        self.GAMMA = Params.GAMMA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        # create the environment
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
        job_log = "../src/simulator/test/filtered-longhorn-v100.log"
        slurm_config = "../src/simulator/test/slurm_config.json"
        backfill_config = "../src/simulator/test/backfill_config.json"

        start_time = datetime.strptime("2021-03-01T00:00:00",
                                       "%Y-%m-%dT%H:%M:%S")  # init the time you want the simulator to start from
        log_start_time = datetime.strptime("2021-03-01T00:00:00",
                                           "%Y-%m-%dT%H:%M:%S")  # init the start time for log sample
        log_end_time = datetime.strptime("2021-07-31T23:59:59", "%Y-%m-%dT%H:%M:%S")  # init the end time for log sample

        self.env = Env(job_log, slurm_config, backfill_config, start_time, log_start_time, log_end_time)

        self.total_rewards = deque([], maxlen=1)

        # flag to figure out if we have render a single episode current epoch
        self.finished_rendering_this_epoch = False

    def solve_environment(self):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        episode = 0
        epoch = 0

        print(f"epoch: {epoch}", flush=True)

        while epoch < self.NUM_EPOCHS:

            # play an episode of the environment
            (sum_of_episode_rewards, episode) = self.play_episode(episode=episode)
            print(f"sum_of_episode_rewards: {sum_of_episode_rewards}", flush=True)
            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # if the epoch is over - we have epoch trajectories to perform the policy gradient
            if episode >= self.BATCH_SIZE:
                # reset the rendering flag
                self.finished_rendering_this_epoch = False

                # reset the episode count
                episode = 0

                # increment the epoch
                epoch += 1

                # feedback
                print(f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                      end="\n", flush=True)

    def play_episode(self, episode: int):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """
        # reset the environment to a random initial state every epoch
        self.env.reset()

        #
        state = self.env.state()

        # initialize the episode arrays
        average_rewards = np.empty(shape=(0,), dtype=float)
        episode_rewards = np.empty(shape=(0,), dtype=float)

        # episode loop
        while True:
            state = np.asarray(state)
            logging.info("state is :{}".format(state.shape))

            # print(f"state: {state}", flush=True)
            if self.env.job_finish():
                action = Categorical(logits=torch.Tensor([[0.0, 1.0]]))
            else:
                action = None

            # take the chosen action, observe the reward and the next state
            # TODO: change the environment to slurm DONE
            state, reward, done = self.env.step(action=action)

            # append the reward to the rewards pool that we collect during the episode
            # we need the rewards so we can calculate the weights for the policy gradient
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)

            # here the average reward is state specific
            average_rewards = np.concatenate((average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)

            # the episode is over
            if done:
                # increment the episode
                episode += 1

                # # calculate the sum of the rewards for the running average metric
                sum_of_rewards = np.sum(episode_rewards)

                # won't render again this epoch
                self.finished_rendering_this_epoch = True

                return sum_of_rewards, episode

    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (
    torch.Tensor, torch.Tensor):
        """
            Calculates the policy "loss" and the entropy bonus
            Args:
                epoch_logits: logits of the policy network we have collected over the epoch
                weighted_log_probs: loP * W of the actions taken
            Returns:
                policy loss + the entropy bonus
                entropy: needed for logging
        """
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.BETA * entropy

        return policy_loss + entropy_bonus, entropy

    @staticmethod
    def get_discounted_rewards(rewards: np.array, gamma: float) -> np.array:
        """
            Calculates the sequence of discounted rewards-to-go.
            Args:
                rewards: the sequence of observed rewards
                gamma: the discount factor
            Returns:
                discounted_rewards: the sequence of the rewards-to-go
        """
        discounted_rewards = np.empty_like(rewards, dtype=np.float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards


def main():
    args = parser.parse_args()
    env = args.env
    use_cuda = args.use_cuda

    random.seed(37)  # 37
    # assert(env in ['CartPole', 'LunarLander'])
    print(f"starting", flush=True)
    baseline = Baseline(problem=env, use_cuda=use_cuda)
    baseline.solve_environment()


if __name__ == "__main__":
    main()
