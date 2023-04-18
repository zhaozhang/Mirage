import os
import re
import argparse
from matplotlib import pyplot as plt
import numpy as np


def plot_fig(reward, file_name):
    x = np.array([x for x in range(0, len(reward))])
    plt.plot(x, reward, 'g', label='Average reward')
    plt.title("Reward Curve")
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-log", required=True)
    parser.add_argument("-out", type=str, default="./avg_reward.png")
    args = parser.parse_args()

    with open(args.log) as f:
        content = f.read()
    content = content.split("\n")
    content_episode = list(filter(lambda x: "Episode" in x, content))
    content_epoch = list(filter(lambda x: "Epoch" in x, content))
    reward_lst = []
    for item in content_episode:
        reward = re.search(r"Reward: ([-+]?[0-9]*\.[0-9]+)", item).group(1)
        reward_lst.append(float(reward))
    epoch_avg_reward = []
    for item in content_epoch:
        reward = re.search(r"Avg Return per Epoch: ([-+]?[0-9]*\.[0-9]+)", item).group(1)
        epoch_avg_reward.append(float(reward))
    # plot_fig(reward_lst, args.out, "reward")
    plot_fig(epoch_avg_reward, args.out)


if __name__ == '__main__':
    main()
