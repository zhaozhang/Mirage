import argparse
from src.sim import simparser
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def plot_bar(short_job, total_job):
    x = np.arange(len(short_job))
    a = np.array(list(short_job.values()))
    b = np.array(list(total_job.values()))

    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.figure(figsize=(12.8, 9.6))
    plt.bar(x, a, width=width, label='short')
    plt.bar(x + width, b, width=width, label='total')
    plt.xticks(np.arange(len(short_job)), list(short_job.keys()), rotation=90)
    plt.xlabel("Year-Month")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-log", type=str, required=True)
    parser.add_argument("-out", default="./")
    parser.add_argument("-pivot", type=float, default=0.01)
    args = parser.parse_args()

    job_dict = simparser.SimParser.parse_job(args.log)
    month_short_dict = {}
    month_total_dict = {}
    for key, value in job_dict.items():
        month_key = datetime.strftime(value.submit, "%Y-%m")
        if value.duration.total_seconds() / (value.time_limit * 60) < args.pivot:
            if month_key not in month_short_dict.keys():
                month_short_dict[month_key] = 1
            else:
                month_short_dict[month_key] += 1
        if month_key not in month_total_dict.keys():
            month_total_dict[month_key] = 1
        else:
            month_total_dict[month_key] += 1

    month_short_dict = dict(sorted(month_short_dict.items()))
    month_total_dict = dict(sorted(month_total_dict.items()))
    plot_bar(month_short_dict, month_total_dict)


if __name__ == '__main__':
    main()
