import torch
import numpy as np
from typing import List
import re
from sklearn import preprocessing
from datetime import timedelta


class parse_state:

    @staticmethod
    def read_state() -> List[torch.Tensor]:
        state = []
        tensors = []
        le_partition = preprocessing.LabelEncoder()
        le_partition.fit(['small', 'normal', 'dev', 'large', 'flex'])
        le_status = preprocessing.LabelEncoder()
        le_status.fit(['PD', 'R', 'CG'])
        with open('interrupt-free-provisioning/policy-gradient/sample_state') as f:
            lines = f.readlines()
            for line in lines[2:]:
                state.append(line)

        for job in state:

            items = job.split()
            # print(items)
            arr = np.zeros(shape=95)
            for i in range(len(items)):
                if i == 0:
                    arr[0] = int(items[i])
                elif i == 1:
                    # partition type
                    arr[1] = le_partition.transform([items[i]])[0]
                elif i == 2 or i == 3:
                    continue
                elif i == 4:
                    arr[2] = le_status.transform([items[i]])[0]
                elif i == 5:
                    days = 0
                    hours = 0
                    minutes = 0
                    seconds = 0
                    times = items[i].split("-")
                    if (len(times) == 2):
                        days = times[0]
                        times = times[1:]
                    times = times[0].split(":")
                    if len(times) == 3:
                        hours = times[0]
                        times = times[1:]
                    minutes = times[0]
                    seconds = times[1]
                    delta = timedelta(days=int(days), seconds=int(seconds), minutes=int(minutes), hours=int(hours))
                    arr[3] = delta.total_seconds()
                elif i == 6:
                    arr[4] = int(items[i])
                elif i == 7:
                    if items[4] == 'R':
                        ranges = re.findall("[0-9]+-[0-9]+", items[7])
                        for val in ranges:
                            start, end = val.split('-')
                            start, end = int(start), int(end)
                            while start <= end:
                                arr[start - 1 + 5] = 1
                                start += 1
                        ranges = re.findall("[0-9]+", items[7])
                        for val in ranges:
                            arr[int(val) - 1 + 5] = 1
            tensor_val = torch.tensor(arr)
            tensors.append(tensor_val)
        return tensors


def main():
    tensors = parse_state.read_state()
    print(tensors)


if __name__ == '__main__':
    main()
