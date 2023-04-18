import argparse
import json
import os
import shutil
import subprocess
import re
from datetime import datetime


def parse_config(config, benchmarks, output_dir):
    with open(config) as f:
        config_json = json.load(f)
    abs_output_dir = os.path.abspath(output_dir)
    os.chdir(output_dir)
    tasks = config_json["tasks"]
    for benchmark in benchmarks:
        tasks[benchmark]["header"]["log"] = os.path.join(abs_output_dir, benchmark,
                                                         os.path.basename(tasks[benchmark]["header"]["log"]))
        try:
            os.makedirs(benchmark, exist_ok=False)
        except OSError:
            print("Directory {} existed: Do you want to overwrite? (y/n)".format(os.path.abspath(benchmark)))
            answer = input()
            while answer != "y" and answer != "n":
                print("Error input. Please enter again.")
                answer = input()
            if answer == "y":
                shutil.rmtree(benchmark)
                os.makedirs(benchmark)
            else:
                continue
        with open(os.path.join(benchmark, "{}_sbatch".format(benchmark)), "w") as f:
            f.write(config_json["general_args"]["script_header"].format(**tasks[benchmark]["header"]))
            f.write("\n")
            f.write(config_json["general_args"][tasks[benchmark]["type"]].format(**tasks[benchmark]["args"]))
        tasks[benchmark]["recent_exec_time"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if tasks[benchmark]["type"] == "train_policy_gradient_str":
            with open(os.path.join(benchmark, "train_cfg.json"), "w") as f:
                json.dump(config_json["general_policy_model_cfg"] | tasks[benchmark]["policy_cfg"], f)
            with open(os.path.join(benchmark, "sim.json"), "w") as f:
                json.dump(config_json["general_policy_sim_cfg"] | tasks[benchmark]["policy_sim_cfg"], f)

    return config_json


def run(script, project_name, clean=False):
    curr_dir = os.getcwd()
    os.chdir(os.path.dirname(script))
    cmd = "sbatch -A {} {}".format(project_name, os.path.basename(script))
    print("Command: {}".format(cmd))
    if not clean:
        temp_str = subprocess.check_output(cmd, shell=True).decode('UTF-8')
        try:
            job_id = re.search(r"Submitted batch job ([0-9]+)", temp_str).group(1)
        except TypeError:
            print("Submit error! Info: {}".format(temp_str))
            exit(1)
        os.chdir(curr_dir)
    else:
        job_id = 0
        os.chdir(curr_dir)
    return job_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-od", "--output_dir", default="./", type=str)
    parser.add_argument("-benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-proj", "--project", type=str, required=True)
    args = parser.parse_args()

    curr_path = os.getcwd()
    os.makedirs(args.output_dir, exist_ok=True)

    new_config = parse_config(args.config, args.benchmarks, args.output_dir)
    job_id_lst = []
    for benchmark in args.benchmarks:
        job_id_lst.append(run(os.path.join(benchmark, "{}_sbatch".format(benchmark)), args.project, False))

    print("Job lists: {}".format(job_id_lst))

    os.chdir(curr_path)
    with open(args.config, "w") as f:
        json.dump(new_config, f, indent=2)


if __name__ == '__main__':
    main()
