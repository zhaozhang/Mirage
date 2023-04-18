import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
import re
import numpy


def plot_fig(expert_name, expert_loss, cluster_name, validation_loss=None):
    x = numpy.array([x for x in range(0, len(expert_loss))])
    expert_loss = list(map(lambda x: numpy.mean(x), expert_loss.values()))
    plt.plot(x, expert_loss, 'g', label='Training loss')
    plt.plot(x, validation_loss, 'b', label='Validation loss')
    plt.title(cluster_name + "_" + expert_name)
    plt.legend()
    plt.savefig(f"{cluster_name}_{expert_name}.png")
    plt.close()


def plot_mix_fig(cluster_name, validation_loss, train_loss):
    x = numpy.array([x for x in range(0, len(validation_loss))])
    train_loss_value = list(map(lambda x: numpy.mean(x), train_loss.values()))
    plt.plot(x, validation_loss, 'b', label='Validation loss')
    plt.plot(x, train_loss_value, 'g', label='Training loss')
    plt.title(cluster_name + "_mixture")
    plt.legend()
    plt.savefig(f"{cluster_name}_mixture.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wd", "--work_dir", required=True)
    args = parser.parse_args()

    os.chdir(args.work_dir)
    for item in os.listdir():
        expert = {}
        mixture_validate = {}
        mixture_train = {}
        with open(item) as f:
            content = f.read()
        content = content.split("\n")
        content_validate = list(
            filter(lambda x: re.match(r"Expert [0-9]+ - MAE after Epoch [0-9]+: .*? hours.", x), content))
        content_train = list(
            filter(lambda x: re.match(r"Expert [0-9]+ - Training loss in Epoch [0-9]+: .*? hours.", x), content))
        content_mix_validate = list(
            filter(lambda x: re.match(r"Mixture of Expert - MAE after Epoch [0-9]+: .*? hours.", x), content))
        content_mix_train = list(
            filter(lambda x: re.match(r"Mixture of Expert - Training loss in Epoch [0-9]+: .*? hours", x), content))
        for train_item in content_train:
            search_number = re.search("Expert ([0-9]+) - Training loss in Epoch ([0-9]+): (.*?) hours.", train_item)
            if search_number.group(1) not in expert.keys():
                expert[search_number.group(1)] = [{}, []]
            if search_number.group(2) not in expert[search_number.group(1)]:
                expert[search_number.group(1)][0][search_number.group(2)] = []
            expert[search_number.group(1)][0][search_number.group(2)].append(float(search_number.group(3)))
        for valid_item in content_validate:
            search_number = re.search("Expert ([0-9]+) - MAE after Epoch ([0-9]+): (.*?) hours.", valid_item)
            expert[search_number.group(1)][1].append(float(search_number.group(3)))
        for key, value in expert.items():
            plot_fig("expert_" + key, value[0], item.split("_")[0], value[1])
        for mix_item in content_mix_validate:
            search_number = re.search("Mixture of Expert - MAE after Epoch ([0-9]+): (.*?) hours.", mix_item)
            mixture_validate[int(search_number.group(1))] = float(search_number.group(2))
        for mix_item in content_mix_train:
            search_number = re.search("Mixture of Expert - Training loss in Epoch ([0-9]+): (.*?) hours", mix_item)
            if int(search_number.group(1)) not in mixture_train.keys():
                mixture_train[int(search_number.group(1))] = []
            mixture_train[int(search_number.group(1))].append(float(search_number.group(2)))
        plot_mix_fig(item.split("_")[0], list(mixture_validate.values()), mixture_train)


if __name__ == '__main__':
    main()
