import json
import os
import shutil
import sys

sys.path.append("../")
sys.path.append("../../")
from moe.predictor import raw_features_to_np_ndarray
import torch
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse
from env import *
from moe.model import StochasticTransformer
from moe.predictor import Predictor
import gc


class Params:
    GAMMA = 0.99  # discount rate
    BETA = 0.1  # the entropy bonus multiplier
    ACTION_SPACE = 2  # submit/no-submit task

    def __init__(self, lr, epoch, batch_size):
        self.ALPHA = lr
        self.NUM_EPOCHS = epoch
        self.BATCH_SIZE = batch_size


def infer_wrapper(episode_actions, episode_logits, average_rewards, episode_rewards, device, agent):
    def infer_transformer(_, data_input):

        np_ndarray_input = raw_features_to_np_ndarray(data_input, parallel=False)

        # From numpy array to pytorch tensor
        tensor_input = torch.from_numpy(np_ndarray_input)
        tensor_input = tensor_input.to(device)
        tensor_input = tensor_input.float()
        tensor_output = agent(tensor_input)
        future_action = Categorical(logits=tensor_output).sample()

        if data_input[0][0][2][3] == 172800.0:
            episode_actions[0] = torch.cat((episode_actions[0], torch.tensor([1]).to(device)), dim=0)
            episode_logits[0] = torch.cat((episode_logits[0], tensor_output), dim=0)
            return 1.0

        episode_actions[0] = torch.cat((episode_actions[0], future_action), dim=0)
        episode_logits[0] = torch.cat((episode_logits[0], tensor_output), dim=0)
        if future_action.item() == 0:
            episode_rewards[0] = np.concatenate((episode_rewards[0], np.array([0.0])), axis=0)
            average_rewards[0] = np.concatenate((average_rewards[0], np.expand_dims(np.mean(episode_rewards), axis=0)),
                                                axis=0)
        return future_action.cpu().detach().item()

    return infer_transformer


class PolicyGradient:
    def __init__(self, train_cfg: Params, use_cuda: bool = False, num_expert: int = 10, base_model_dir: str = "./",
                 sim_config: dict = None, base_model_name: str = "transformer", model_output: str = "tx_policy",
                 base_lr: float = 1e-5, base_model_type: str = "transformer", base_out_dim: int = 1):

        self.NUM_EPOCHS = train_cfg.NUM_EPOCHS
        self.ALPHA = train_cfg.ALPHA
        self.BATCH_SIZE = train_cfg.BATCH_SIZE
        self.GAMMA = Params.GAMMA
        self.BETA = Params.BETA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

        # specify dimensions of state, action, and hidden spaces
        self.ACTION_SPACE_SIZE = Params.ACTION_SPACE

        # instantiate the tensorboard writer
        self.writer = SummaryWriter(comment=f'_PG_CP_Gamma={self.GAMMA},'
                                            f'LR={self.ALPHA},'
                                            f'BS={self.BATCH_SIZE},'
                                            f'BETA={self.BETA}')

        # create the environment
        # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

        self.env = Env(**sim_config)
        self.model_output = os.path.join(base_model_dir, model_output)
        self.num_expert = num_expert

        model_path = os.path.join(base_model_dir, f"{base_model_name}.pt")
        print(f'Loading base model: {model_path}')
        checkpoint = torch.load(model_path)
        base_model_name = checkpoint['model_name']
        model_hparams = checkpoint['model_hparams']
        model_state = checkpoint['model_state_dict']
        optimizer_name = checkpoint['optimizer_name']
        optimizer_state = checkpoint['optimizer_state_dict']
        base_model_predictor = Predictor(model_name=base_model_name, hparams=model_hparams,
                                         optimizer_name=optimizer_name,
                                         model_state=model_state, optimizer_state=optimizer_state,
                                         base_learning_rate=base_lr,
                                         device="gpu" if torch.cuda.is_available() and use_cuda else "cpu")

        self.base_model = base_model_predictor.model
        if base_model_type != "policy-gradient":
            self.agent = StochasticTransformer(base_out_dim, self.base_model)
            self.adam = optim.Adam(params=self.agent.parameters(), lr=self.ALPHA)
        else:
            self.agent = self.base_model
            self.adam = base_model_predictor.optimizer
        self.agent.to(self.DEVICE)
        print("----------------------------------------------------------")
        print("Current model: ")
        print(self.agent)
        print("----------------------------------------------------------")
        print(
            "Current parameters: lr: {}, batch_size: {}, epoch:{}".format(self.ALPHA, self.BATCH_SIZE, self.NUM_EPOCHS))
        print("----------------------------------------------------------")

        self.total_rewards = deque([], maxlen=100)

    def solve_environment(self, init_epoch=0, save_checkpoint=-1):
        """
            The main interface for the Policy Gradient solver
        """
        # init the episode and the epoch
        episode = 0
        epoch = init_epoch

        # init the epoch arrays
        # used for entropy calculation
        epoch_logits = torch.empty(size=(0, self.ACTION_SPACE_SIZE), device=self.DEVICE)
        epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

        while epoch < self.NUM_EPOCHS:

            # play an episode of the environment
            (episode_weighted_log_prob_trajectory,
             episode_logits,
             sum_of_episode_rewards,
             episode) = self.play_episode(episode=episode)

            # after each episode append the sum of total rewards to the deque
            self.total_rewards.append(sum_of_episode_rewards)

            # append the weighted log-probabilities of actions
            epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),
                                                 dim=0)

            # append the logits - needed for the entropy bonus calculation
            epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

            # if the epoch is over - we have epoch trajectories to perform the policy gradient
            if episode >= self.BATCH_SIZE:

                # reset the episode count
                episode = 0

                # increment the epoch
                epoch += 1

                # calculate the loss
                loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,
                                                    weighted_log_probs=epoch_weighted_log_probs)

                # zero the gradient
                self.adam.zero_grad()

                # backprop
                loss.backward()

                # update the parameters
                self.adam.step()

                # feedback
                print(f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}", flush=True)

                self.writer.add_scalar(tag='Average Return over 100 episodes',
                                       scalar_value=np.mean(self.total_rewards),
                                       global_step=epoch)

                self.writer.add_scalar(tag='Entropy',
                                       scalar_value=entropy,
                                       global_step=epoch)

                # reset the epoch arrays
                # used for entropy calculation
                epoch_logits = torch.empty(size=(0, self.ACTION_SPACE_SIZE), device=self.DEVICE)
                epoch_weighted_log_probs = torch.empty(size=(0,), dtype=torch.float, device=self.DEVICE)

                file_name = "{}_{}.pt".format(self.model_output, epoch)
                if save_checkpoint == epoch or save_checkpoint == -1:
                    try:
                        torch.save({
                            'model_name': "transformer_policy",
                            'model_hparams': {'in_dim': self.agent.indim, 'base': self.agent.base_model[0]},
                            'model_state_dict': self.agent.state_dict(),
                            'optimizer_name': "Adam",
                            'optimizer_state_dict': self.adam.state_dict(),
                        }, os.path.join(file_name))
                        print(f"Model transformer_policy is successfully checkpointed to {file_name}")
                    except:
                        print(f'Error: Failed checkpointing to path: {file_name}!')

                del loss
                del entropy
                gc.collect()
                torch.cuda.empty_cache()

                # check if solved
                if np.mean(self.total_rewards) > 200:
                    print('\nSolved!')
                    break

        # close the writer
        self.writer.close()

    def play_episode(self, episode: int):
        """
            Plays an episode of the environment.
            episode: the episode counter
            Returns:
                sum_weighted_log_probs: the sum of the log-prob of an action multiplied by the reward-to-go from that state
                episode_logits: the logits of every step of the episode - needed to compute entropy for entropy bonus
                finished_rendering_this_epoch: pass-through rendering flag
                sum_of_rewards: sum of the rewards for the episode - needed for the average over 200 episode statistic
        """

        # reset the environment to a random initial state every epoch
        self.env.reset()

        # initialize the episode arrays
        episode_actions = [torch.empty(size=(0,), dtype=torch.long, device=self.DEVICE)]
        episode_logits = [torch.empty(size=(0, self.ACTION_SPACE_SIZE), device=self.DEVICE)]
        average_rewards = [np.empty(shape=(0,), dtype=float)]
        episode_rewards = [np.empty(shape=(0,), dtype=float)]

        # episode loop
        # take the chosen action, observe the reward and the next state
        reward, interruption_overlap = self.env.reward(
            infer_wrapper(episode_actions, episode_logits, average_rewards, episode_rewards, self.DEVICE,
                          self.agent))

        print(f"Episode: {episode + 1}, "
              f"Reward: {reward}, "
              f"Simulation Time: {self.env.sim_start_time}, "
              f"Interruption/Overlap: {interruption_overlap}, "
              f"Inference Times: {episode_rewards[0].size}", flush=True)

        # append the reward to the rewards pool that we collect during the episode
        # we need the rewards so we can calculate the weights for the policy gradient
        # and the baseline of average
        episode_rewards = np.concatenate((episode_rewards[0], np.array([reward])), axis=0)

        # here the average reward is state specific
        average_rewards = np.concatenate((average_rewards[0],
                                          np.expand_dims(np.mean(episode_rewards), axis=0)),
                                         axis=0)
        episode_actions = episode_actions[0]
        episode_logits = episode_logits[0]

        # increment the episode
        episode += 1

        # turn the rewards we accumulated during the episode into the rewards-to-go:
        # earlier actions are responsible for more rewards than the later taken actions
        discounted_rewards_to_go = PolicyGradient.get_discounted_rewards(rewards=episode_rewards,
                                                                         gamma=self.GAMMA)
        discounted_rewards_to_go -= average_rewards  # baseline - state specific average

        # # calculate the sum of the rewards for the running average metric
        sum_of_rewards = np.sum(episode_rewards)

        # set the mask for the actions taken in the episode
        mask = one_hot(episode_actions, num_classes=self.ACTION_SPACE_SIZE)

        # calculate the log-probabilities of the taken actions
        # mask is needed to filter out log-probabilities of not related logits
        episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)

        # weight the episode log-probabilities by the rewards-to-go
        episode_weighted_log_probs = episode_log_probs * \
                                     torch.tensor(discounted_rewards_to_go).float().to(self.DEVICE)

        # calculate the sum over trajectory of the weighted log-probabilities
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        return sum_weighted_log_probs, episode_logits, sum_of_rewards, episode

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
        discounted_rewards = np.empty_like(rewards, dtype=float)
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=gamma)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
        return discounted_rewards


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', help='Use if you want to use CUDA', action='store_true')
    parser.add_argument("-config", required=True, help="Simulation config JSON file")
    parser.add_argument("-base_dir", required=True, help="Base model directory")
    parser.add_argument("-output_dir", required=True, help="Output directory")
    parser.add_argument("-init_epoch", type=int, default=0, help="Initial epoch number")
    parser.add_argument("-base_prefix", required=True, help="The prefix of base model file name")
    parser.add_argument("-base_lr", type=float, default=1e-5, help="The learning rate of base model")
    parser.add_argument("-base_type", type=str, default="transformer", help="The type of base model")
    parser.add_argument("-base_out_dim", type=int, default=1,
                        help="The output dimension of base model (Can be ignored if resume training)")
    parser.add_argument("-train_cfg", required=True, help="The config of training parameter")
    parser.add_argument("-save", default=-1, type=int, help="Save checkpoint at specific epoch")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    with open(args.train_cfg) as f:
        train_cfg = json.load(f)
    train_cfg = Params(**train_cfg)

    if os.path.abspath(args.base_dir) != os.path.abspath(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(os.path.join(args.base_dir, "{}.pt".format(args.base_prefix)),
                    os.path.join(args.output_dir, "{}.pt".format(args.base_prefix)))

    policy_gradient = PolicyGradient(train_cfg=train_cfg, use_cuda=args.use_cuda, num_expert=1,
                                     base_model_dir=args.output_dir, base_model_name=args.base_prefix,
                                     sim_config=config, base_model_type=args.base_type, base_lr=args.base_lr,
                                     base_out_dim=args.base_out_dim)
    policy_gradient.solve_environment(args.init_epoch, args.save)


if __name__ == "__main__":
    main()
