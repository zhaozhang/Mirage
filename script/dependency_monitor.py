import os
from subprocess import check_output
import argparse
import time
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-samples", type=int, required=True)
    parser.add_argument("-jobid", required=True)
    parser.add_argument("-od", default=".")
    args = parser.parse_args()

    collection = []

    for _ in range(0, args.samples):
        out = str(check_output(f"sacct --format=jobid,JobName,UID,Priority -j {args.jobid}", shell=True))
        last_line = out.split("\\n")[-2]
        priority = list(filter(lambda x: x != "", last_line.split(" ")))[-1]
        collection.append(int(priority))
        time.sleep(600)

    with open(os.path.join(args.od, "monitor.pickle"), "wb") as f:
        pickle.dump(collection, f)


if __name__ == '__main__':
    main()
