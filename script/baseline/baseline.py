import json
import pickle
import numpy as np


def main():
    with open('test.pickle', 'rb') as f:
        content = pickle.load(f)
        counter = 0
        reward_min = []
        reward_max = []
        for item in content:
            if counter == 0:
                reward_min.append(item[1] / 3600)
            elif counter == 4:
                reward_max.append(item[1] / 3600)
            counter += 1
            if counter == 5:
                counter = 0
        reward_min = np.array(reward_min)
        reward_max = np.array(reward_max)
        with open('reward_max.json', 'w') as f:
            json.dump(reward_max.tolist(), f)
        print(np.percentile(reward_max, 100))
        print(np.percentile(reward_max, 90))
        print(np.percentile(reward_max, 75))
        print(np.percentile(reward_max, 50))
        print(np.percentile(reward_max, 25))
        print(np.percentile(reward_max, 0))
        print(np.mean(reward_max))

        print(np.percentile(reward_min, 100))
        print(np.percentile(reward_min, 90))
        print(np.percentile(reward_min, 75))
        print(np.percentile(reward_min, 50))
        print(np.percentile(reward_min, 25))
        print(np.percentile(reward_min, 0))
        print(np.mean(reward_min))


if __name__ == '__main__':
    main()
