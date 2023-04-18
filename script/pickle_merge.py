import pickle
import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wd", "--work_dir", required=True)
    parser.add_argument("-file", default=None)
    parser.add_argument("-out", "--output_name", required=True)
    parser.add_argument("-key", default="data")
    parser.add_argument("-simple_merge", action="store_true", default=False)
    args = parser.parse_args()

    os.chdir(args.work_dir)

    files = os.listdir()
    files = list(filter(lambda x: re.match(rf"{args.key}_[0-9]+.pickle", x), files))
    files = list(sorted(files))
    print(files)

    if args.simple_merge:
        content = []
        for item in files:
            with open(item, "rb") as f:
                content += pickle.load(f)
        with open(args.output_name, "wb") as f:
            pickle.dump(content, f)
    else:
        content = {}

        if args.file is None:
            for item in files:
                with open(item, "rb") as f:
                    temp = pickle.load(f)
                content.update(temp)
        else:
            with open(args.file, "rb") as f:
                temp = pickle.load(f)
            content.update(temp)

        content = dict(sorted(content.items()))

        content_out = []
        for key, value in content.items():
            content_out += value

        with open(args.output_name, "wb") as f:
            pickle.dump(content_out, f)


if __name__ == '__main__':
    main()
