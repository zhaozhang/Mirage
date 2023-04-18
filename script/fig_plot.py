import numpy as np
from matplotlib import pyplot as plt
import pickle
import os
import argparse
import json
from datetime import datetime
from datetime import timedelta


def cumulative_frequency(data, nbins):
    hist, bins = np.histogram(a=data, bins=nbins, density=True, range=(min(data), max(data)))
    hist = np.insert(hist, 0, 0)
    hist /= np.sum(hist)
    cum_hist = np.cumsum(hist)
    assert (round(cum_hist[-1]) == 1.0)
    return bins, cum_hist


def backfill_report(backfill_data, config, workload_name, file, skip=None):
    with open(file, "a+") as f:
        backfill_invalid_time = datetime.strptime("2077-01-01T01:00:00", "%Y-%m-%dT%H:%M:%S")
        for model_type in config["validation"]:
            for key, value in backfill_data.items():
                if skip is not None and key in skip:
                    continue
                f.write("-----------------------------------------------------------------\n")
                f.write(f"{key} workload ({workload_name} {model_type} backfill cases)\n")
                f.write("{:<10}{:<30}{:<30}{:<10}{:<10}{:<30}\n".format("Overlap", "Backfill Time", "Submission Time",
                                                                        "Index", "Pred", "Sample Start Time"))
                count_backfill = 0
                count_backfill_large_overlap = 0
                for item in value[model_type]:
                    if item[1] != backfill_invalid_time:
                        count_backfill += 1
                    if item[0] <= -10:
                        if item[1] != backfill_invalid_time:
                            count_backfill_large_overlap += 1
                        workload_start_time = datetime.strptime("2021-03-01T00:00:00", "%Y-%m-%dT%H:%M:%S") + timedelta(
                            hours=item[3] * 4)
                        f.write(
                            f"{item[0]:<10.2f}{str(item[1]):<30}{str(item[2]):<30}{str(item[3]):<10}{item[4]:<10.2f}{str(workload_start_time):<30}\n")
                f.write(
                    f"{count_backfill}, {count_backfill_large_overlap}, {count_backfill_large_overlap / count_backfill}\n")
                f.write("-----------------------------------------------------------------\n")


def draw(data, name, title, config, filter_index, xmin=-20, xmax=80, nbins=50, output_dir="./", mode="CDF"):
    box_plot_lst = []
    box_x_tick = []
    for item in config["baseline"]:
        filtered_data = []
        for i in range(0, len(data[item])):
            if i not in filter_index:
                filtered_data.append(data[item][i])
        if mode == "CDF":
            x, y = cumulative_frequency(filtered_data, nbins=nbins)
            plt.plot(x, y, label=item)
        else:
            box_plot_lst.append(filtered_data)
            box_x_tick.append(item.replace("baseline_", ""))
    for item in config["validation"]:
        filtered_data = []
        for i in range(0, len(data[item])):
            if i not in filter_index:
                filtered_data.append(data[item][i])
        if mode == "CDF":
            x, y = cumulative_frequency(filtered_data, nbins=nbins)
            plt.plot(x, y, label=item)
        else:
            box_plot_lst.append(filtered_data)
            box_x_tick.append(item.replace("validation_", ""))

    if mode == "CDF":
        plt.xlim(xmin, xmax)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Interrupt/Overlap')
        plt.ylabel('Cumulative Probability')
        plt.legend()
    else:
        bp_dict = plt.boxplot(box_plot_lst, showmeans=True, widths=0.3)
        plt.gcf().autofmt_xdate()
        plt.rcParams.update({'font.size': 10})
        plt.xticks(range(1, len(box_x_tick) + 1), box_x_tick, fontsize=10)
        plt.ylabel('Interrupt/Overlap (Hours)')
        median = []
        mean = []
        for item in box_plot_lst:
            median.append(np.median(item))
        median = np.array(median)
        for item in box_plot_lst:
            mean.append(np.mean(item))
        mean = np.array(mean)
        for i, line in enumerate(bp_dict['medians']):
            # get position data for median line
            x, y = line.get_xydata()[1]  # top of median line
            # overlay median value
            plt.text(x, y, ' mean=%.1f\n med=%.1f' % (mean[i], median[i]))  # draw above, centered


        # for line in bp_dict['boxes']:
        #     x, y = line.get_xydata()[0]  # bottom of left line
        #     plt.text(x, y, '%.1f' % x,
        #              horizontalalignment='center',  # centered
        #              verticalalignment='top')  # below
        #     x, y = line.get_xydata()[3]  # bottom of right line
        #     plt.text(x, y, '%.1f' % x,
        #              horizontalalignment='center',  # centered
        #              verticalalignment='bottom')  # below

    fig = plt.gcf()
    fig.set_size_inches(12.5, 6.5)
    plt.title(title)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{name}.png"), dpi=100)
    plt.close()


def classify_backfill(data: {}, backfill_new, new_model_name):
    backfill_invalid_time = datetime.strptime("2077-01-01T01:00:00", "%Y-%m-%dT%H:%M:%S")
    data_backfill = {}
    data_no_backfill = {}
    for i in range(0, len(backfill_new[new_model_name])):
        if backfill_new[new_model_name][i][0] == backfill_invalid_time:
            for key, value in data.items():
                if key not in data_no_backfill.keys():
                    data_no_backfill[key] = [value[i]]
                else:
                    data_no_backfill[key].append(value[i])
        else:
            for key, value in data.items():
                if key not in data_backfill.keys():
                    data_backfill[key] = [value[i]]
                else:
                    data_backfill[key].append(value[i])
    return data_no_backfill, data_backfill


def classify(data, pivot, backfill_data, config, filter_valid=True):
    backfill_invalid_time = datetime.strptime("2077-01-01T01:00:00", "%Y-%m-%dT%H:%M:%S")
    new_dict = {"low": {}, "medium": {}, "heavy": {}}
    new_backfill = {"low": {}, "medium": {}, "heavy": {}}
    filter_index = {"low": [], "medium": [], "heavy": []}
    for i in range(0, len(data[pivot])):
        if data[pivot][i] <= 2:
            target = "low"
        elif 2 < data[pivot][i] <= 12:
            target = "medium"
        else:
            target = "heavy"
        for key, value in data.items():
            if key in config["validation"]:
                if key not in new_backfill[target].keys():
                    new_backfill[target][key] = [
                        [value[i], backfill_data[key][i][0], backfill_data[key][i][1], backfill_data[key][i][2],
                         backfill_data[key][i][3]]]
                else:
                    new_backfill[target][key].append(
                        [value[i], backfill_data[key][i][0], backfill_data[key][i][1], backfill_data[key][i][2],
                         backfill_data[key][i][3]])
                if value[i] <= -10 and backfill_invalid_time != backfill_data[key][i][0]:
                    filter_index[target].append(len(new_backfill[target][key]) - 1)
            if key not in new_dict[target].keys():
                new_dict[target][key] = [value[i]]
            else:
                new_dict[target][key].append(value[i])
    if filter_valid:
        filter_index["low"] = set(filter_index["low"])
        filter_index["medium"] = set(filter_index["medium"])
        filter_index["heavy"] = set(filter_index["heavy"])
    else:
        filter_index["low"].clear()
        filter_index["medium"].clear()
        filter_index["heavy"].clear()

    return new_dict, new_backfill, filter_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-json", required=True)
    parser.add_argument("-wd", "--work_dir", default="./")
    parser.add_argument("-od", "--output_dir", default="./")
    parser.add_argument("-mode", "--plot_mode", default="CDF")
    parser.add_argument("-ref", "--ref_baseline", type=str, default="baseline_reactive")
    parser.add_argument("-filter", action="store_true", default=False)
    parser.add_argument("-backfill_split", action="store_true", default=False)
    parser.add_argument("-sample_max", type=int)
    args = parser.parse_args()

    with open(args.json) as f:
        config = json.load(f)

    os.chdir(args.work_dir)
    output_dir = os.path.join(os.getcwd(), config["fig_output"])
    for item in config["workload"]:
        os.chdir(item)
        data = {}
        validation_backfill = {}
        for baseline in config["baseline"]:
            data[baseline] = []
            with open(os.path.join(baseline, f"{baseline}_{item}_merge.pickle"), "rb") as f:
                content = pickle.load(f)
                for i, sample in enumerate(content):
                    if args.sample_max is not None and i >= args.sample_max:
                        break
                    data[baseline].append(sample[1] / 3600)
        for validation in config["validation"]:
            data[validation] = []
            validation_backfill[validation] = []
            with open(os.path.join(validation, "reward.pickle"), "rb") as f:
                content = pickle.load(f)
                for i, (group, sample) in enumerate(content.items()):
                    if args.sample_max is not None and i >= args.sample_max:
                        break
                    data[validation].append(sample[0][3])
                    validation_backfill[validation].append([sample[0][5],
                                                            datetime.strptime(sample[0][6], "%Y-%m-%d %H:%M:%S"), group,
                                                            sample[0][2]])

        if args.backfill_split:
            data_no_backfill, data_backfill = classify_backfill(data, validation_backfill, config["validation"][0])
            draw(data_no_backfill, item + "_no_backfill", f"No backfill ({item})", config, [], -50, 10,
                 output_dir=output_dir,
                 mode=args.plot_mode)
            draw(data_backfill, item + "backfill", f"Backfill ({item})", config, [], -50, 10, output_dir=output_dir,
                 mode=args.plot_mode)
            os.chdir("../")

        else:
            data, backfill_data, filter_index = classify(data, args.ref_baseline, validation_backfill, config,
                                                         args.filter)

            draw(data["low"], "low_workload_" + item, f"Low workload ({item}, X <= 2)", config, filter_index["low"],
                 -50, 10, output_dir=output_dir, mode=args.plot_mode)
            draw(data["medium"], "medium_workload_" + item, f"Medium workload ({item}, 2 < X <= 12)", config,
                 filter_index["medium"], -30, 30, output_dir=output_dir, mode=args.plot_mode)
            draw(data["heavy"], "heavy_workload_" + item, f"Heavy workload ({item}, X > 12)", config,
                 filter_index["heavy"], -40, 80, 20, output_dir=output_dir, mode=args.plot_mode)
            os.chdir("../")

            backfill_report(backfill_data, config, item, os.path.join(args.output_dir, "backfill_report.log"), ["low"])


if __name__ == '__main__':
    main()
