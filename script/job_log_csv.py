import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--log", required=True, help="The job log")
    parser.add_argument("-o", "--output_csv", required=True, help="The output csv file")
    parser.add_argument("-s", "--start", default="2018-12-04T10:24:58", help="The start time of the job interval")
    parser.add_argument("-e", "--end", default="2022-08-19T23:58:36", help="The end time of the job interval")
    args = parser.parse_args()

    with open(args.log) as f:
        content = f.read()

    table = content.strip().split("\n")
    table_title = list(filter(lambda x: x != "", table[0].split(" ")))
    table_data = table[2:]
    with open(args.output_csv, "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(table_title)
        for item in table_data:
            data_lst = list(filter(lambda x: x != "", item.split(" ")))
            if "_" in data_lst[0]:
                job_id_tmp = data_lst[0].split("_")
                if job_id_tmp[1].isdigit():
                    data_lst[0] = str(int(job_id_tmp[0]) + int(job_id_tmp[1]) - 1)
                else:
                    if job_id_tmp[1][1].isdigit():
                        data_lst[0] = str(int(job_id_tmp[0]) + int(job_id_tmp[1][1]) - 1)
                    else:
                        data_lst[0] = job_id_tmp[0]
            if args.start <= data_lst[4] < args.end:
                csv_writer.writerow(data_lst)


if __name__ == '__main__':
    main()
