import numpy as np
import csv
from matplotlib import pyplot as plt
import pickle

def main():
    with open("../../temp/online_validation/2000/prediction_trace.tickle", "rb") as f:
        content = pickle.load(f)
    expect_reward = []
    actual_reward = []
    error = []
    with open("../../temp/online_validation/2000/reward.log") as f:
        content = csv.reader(f)
        count = 0
        content_tmp = []
        for row in content:
            content_tmp.append(row)
        for item in content_tmp[2:-1]:
            expect_reward.append(float(item[2]))
            actual_reward.append(float(item[3]))
            error.append(float(item[3]) - float(item[2]))
    expect_reward = np.array(expect_reward)
    actual_reward = np.array(actual_reward)
    error = np.array(error)
    print(np.mean(error))
    print(np.mean(actual_reward))
    # actual_reward = np.sort(actual_reward)
    print(np.percentile(actual_reward, 100))
    print(np.percentile(actual_reward, 90))
    print(np.percentile(actual_reward, 75))
    print(np.percentile(actual_reward, 50))
    print(np.percentile(actual_reward, 25))
    print(np.percentile(actual_reward, 0))
    x = np.arange(0, 540, 1)
    #plt.plot(x, expect_reward, label="expected reward")
    plt.scatter(x, actual_reward, label="actual reward", s=2)
    #plt.plot(x, error, label="error")
    #plt.ylim(-40, 40)
    plt.xlim(0, 540)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
