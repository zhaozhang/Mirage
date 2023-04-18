# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import copy
import os, sys
from datetime import datetime
from datetime  import timedelta
from operator import itemgetter
from pprint import pprint
from itertools import chain

import pickle
import numpy as np
import pandas as pd
import ray


DRIVE_ROOT_PATH='D:'
DATA_HOME=os.path.join(DRIVE_ROOT_PATH, 'slurm')
JOB_COMPLETION_LOG_PATH=os.path.join(DATA_HOME,'jobcomp.log')

SQUEUE_LOG_PATH=os.path.join(DATA_HOME,'squeue.out')
SQUEUE_LOG_SEEK_START=950000000
SQUEUE_LOG_SEEK_END=1100000000
SQUEUE_LOG_SEP_LINE= '###############################################################################\n'
SQUEUE_LOG_VALID_RECORD_MIN_NLINES=3
SQUEUE_LOG_PARTITION_SIZE=128


SQUEUE_DATE_TIME_FORMAT='%a %b %d %H:%M:%S %Y'
SQUEUE_RECORD_COLUMNS= 'JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)'
LIST_SQUEUE_RECORD_COLS=SQUEUE_RECORD_COLUMNS.split(' ')
LIST_SQUEUE_RECORD_COLS=[col for col in LIST_SQUEUE_RECORD_COLS if col != '']
IDX_SQUEUE_RECORD_COLS_JOBID=LIST_SQUEUE_RECORD_COLS.index('JOBID')
IDX_SQUEUE_RECORD_COLS_STATUS=LIST_SQUEUE_RECORD_COLS.index('ST')
IDX_SQUEUE_RECORD_COLS_NNODES=LIST_SQUEUE_RECORD_COLS.index('NODES')
IDX_SQUEUE_RECORD_COLS_NODELIST=LIST_SQUEUE_RECORD_COLS.index('NODELIST(REASON)')

JOBCOMP_DATE_TIME_FORMAT= '%Y-%m-%dT%H:%M:%S'

NHOURS_PADDING=24*3+1
TIME_RANGE_PADDING=timedelta(hours=NHOURS_PADDING)

SLURM_JOB_QOS_CODE={
    'normal': 0
}

SLURM_JOB_PARTITION_CODE={
    'normal': 0
}

TIME_WINDOW_STEP_SIZE=0.2
TIME_WINDOW_NUM_STEPS=128
assert(TIME_WINDOW_NUM_STEPS*TIME_WINDOW_STEP_SIZE<=NHOURS_PADDING)
TIME_WINDOW_IMPUTATION=1


@ray.remote
def squeue_partition_parser(partition):
    gather=[]
    for rec in partition:
        res=squeue_record_parser(rec)
        if(res is not None):
            gather.append(res)
    return gather

def squeue_record_parser(record, debug=False):
    str_datetime=record[0]
    datetime_obj = datetime.strptime(str_datetime, SQUEUE_DATE_TIME_FORMAT).replace(second=0, microsecond=0)

    if(debug):
        print(f'Parsing log record (Timestamp: {datetime_obj})')

    str_columns=record[1]
    if(str_columns!=SQUEUE_RECORD_COLUMNS):
        return None

    list_str_jobs=record[2:]
    if(len(list_str_jobs)<=0):
        return None

    pending_jobs=[]
    running_jobs=[]
    for str_job in list_str_jobs:
        if(str_job!=''):
            str_job_log=str_job.split(' ')
            str_job_log= [s for s in str_job_log if s!='']
            if(len(str_job_log)!=len(LIST_SQUEUE_RECORD_COLS)):
                return None

            jobid = str_job_log[IDX_SQUEUE_RECORD_COLS_JOBID]
            jobnnodes=int(str_job_log[IDX_SQUEUE_RECORD_COLS_NNODES])
            jobnodelist=str_job_log[IDX_SQUEUE_RECORD_COLS_NODELIST]

            if( ('PartitionConfig)') in jobnodelist or (jobnnodes<=0)):
                continue

            if(str_job_log[IDX_SQUEUE_RECORD_COLS_STATUS]== 'R'):
                running_jobs.append((jobid, jobnnodes))
            elif(str_job_log[IDX_SQUEUE_RECORD_COLS_STATUS] == 'PD'):
                pending_jobs.append((jobid, jobnnodes))
            else:
                print(f'Unknown job status: {str_job_log[IDX_SQUEUE_RECORD_COLS_STATUS]}')
                return None

    return (datetime_obj, running_jobs, pending_jobs)


def squeue_log_load(log_path, seek_start, seek_end, valid_record_min_nlines, debug=False):
    list_log_records = []

    with open(log_path, 'r') as f_queue_out:
        f_queue_out.seek(seek_start)
        seek_pos = seek_start

        log_record = ''
        log_record_nlines = 0
        log_head_truncate = False
        while (True):
            line = f_queue_out.readline()
            if (line == SQUEUE_LOG_SEP_LINE):
                if (log_record != ''):
                    if (log_record_nlines >= valid_record_min_nlines and log_head_truncate):
                        log_record=log_record.split('\n')
                        list_log_records.append(log_record)
                        if(debug):
                            print(f'Squeue event: {log_record}')
                log_head_truncate = True
                log_record = ''
                log_record_nlines = 0
            else:
                if (line.startswith('t:')):
                    line = line[len('t:'):]
                line = line.strip(' ')
                log_record += line
                log_record_nlines += 1

            seek_pos += len(line)
            if seek_pos > seek_end:
                break
    return list_log_records


def partition_list(lst, size):
    partitions=[]
    for cursor in range(0, len(lst), size):
        partitions.append(lst[cursor:cursor + size])
    return partitions

def coalesce_lists(ls):
    return list(chain.from_iterable(ls))


def jobcmp_strptime_timelimit(strtime):
    if('-' in strtime):
        ndays_limit=int(strtime.split('-')[0])
        within_day_limit=strtime.split('-')[1]
    else:
        ndays_limit=0
        within_day_limit=strtime
    within_day_limit=datetime.strptime(within_day_limit,'%H:%M:%S')
    within_day_limit=timedelta(hours=within_day_limit.hour, minutes=within_day_limit.minute, seconds=within_day_limit.second)

    time_limit_seconds=within_day_limit.total_seconds()+ndays_limit*24*60*60
    time_limit_hours=time_limit_seconds/(60*60)

    return time_limit_hours



def jobcomp_log_load(log_path, debug=True):
    df_job_comp_logs=pd.read_csv(log_path, sep='|')
    df_job_comp_logs=df_job_comp_logs[['JobID',
                                       'Submit',
                                       'Start',
                                       'End',
                                       'NNodes',
                                       'Timelimit',
                                       'QOS',
                                       'Partition',
                                       ]]

    df_job_comp_logs['JobID'] = df_job_comp_logs['JobID'].astype(str)
    df_job_comp_logs['NNodes'] = df_job_comp_logs['NNodes'].astype(int)

    df_job_comp_logs['Submit']=df_job_comp_logs['Submit'].apply(lambda strtime: datetime.strptime(strtime, JOBCOMP_DATE_TIME_FORMAT))
    df_job_comp_logs['Start']=df_job_comp_logs['Start'].apply(lambda strtime: datetime.strptime(strtime, JOBCOMP_DATE_TIME_FORMAT))
    df_job_comp_logs['End']=df_job_comp_logs['End'].apply(lambda strtime: datetime.strptime(strtime, JOBCOMP_DATE_TIME_FORMAT))
    df_job_comp_logs['Timelimit']=df_job_comp_logs['Timelimit'].apply(lambda strtime: jobcmp_strptime_timelimit(strtime))

    df_job_comp_logs['QueueingDelay'] = df_job_comp_logs['Start']-df_job_comp_logs['Submit']
    df_job_comp_logs['QueueingDelay'] = df_job_comp_logs['QueueingDelay'].apply(lambda delta: delta.total_seconds()/3600)
    df_job_comp_logs['CompletionTime'] = df_job_comp_logs['End']-df_job_comp_logs['Start']
    df_job_comp_logs['CompletionTime'] = df_job_comp_logs['CompletionTime'].apply(lambda delta: delta.total_seconds()/3600)

    df_job_comp_logs['Submit']=df_job_comp_logs['Submit'].apply(lambda obj_datetime: obj_datetime.replace(second=0, microsecond=0))
    df_job_comp_logs['Start'] = df_job_comp_logs['Start'].apply(lambda obj_datetime: obj_datetime.replace(second=0, microsecond=0))
    df_job_comp_logs['End'] = df_job_comp_logs['End'].apply(lambda obj_datetime: obj_datetime.replace(second=0, microsecond=0))

    df_job_comp_logs['QOS']=df_job_comp_logs['QOS'].apply(lambda str_qos: SLURM_JOB_QOS_CODE[str_qos])
    df_job_comp_logs['Partition']=df_job_comp_logs['Partition'].apply(lambda str_partition: SLURM_JOB_PARTITION_CODE[str_partition])

    if(debug):
        print('Loading job completion log as pandas df...')
        print(f'Dataframe columns: {df_job_comp_logs.columns}')
        print(f'Example row:\n {df_job_comp_logs.loc[0]}')

    df_job_comp_logs = df_job_comp_logs.drop_duplicates(subset=['JobID'], keep='first')
    df_job_comp_logs=df_job_comp_logs.set_index('JobID')
    df_job_comp_logs=df_job_comp_logs.to_dict('index')

    return df_job_comp_logs

@ray.remote
def squeue_partition_extract_features(partition, job_comp_log_dict):
    partition_feature_mat=[]
    for record in partition:
        features=squeue_record_extract_features(record, job_comp_log_dict)
        partition_feature_mat.append(features)
    return partition_feature_mat


def impute_nan_mean(lst):
    lst_nan_imputed=np.array(lst)
    if(len(lst_nan_imputed))>0:
        if(all(np.isnan(lst_nan_imputed))):
            lst_nan_imputed = np.zeros(len(lst_nan_imputed))
        lst_nan_imputed[np.where(np.isnan(lst_nan_imputed))] = np.nanmean(lst_nan_imputed)
    return lst_nan_imputed

def squeue_record_extract_features(record, job_comp_log_dict, debug=True):
    (timestamp, rjobs, pdjobs)=record

    pdnodes=[]
    pdtime_limits=[]
    pdtime_queue = []
    pdtime_run = []
    for jobid, nnodes in pdjobs:
        pdnodes.append(nnodes)
        if(jobid in job_comp_log_dict):
            pdtime_limits.append(job_comp_log_dict[jobid]['Timelimit'])

            queuetime = timestamp - job_comp_log_dict[jobid]['Submit']
            queuetime_numeric = max(0.0, queuetime.total_seconds() / (60 * 60))
            pdtime_queue.append(queuetime_numeric)

            runtime_numeric=0.0
            pdtime_run.append(runtime_numeric)
        else:
            if(debug):
                print(f'Warning: {timestamp} - pending job {jobid} is not included in jobcomplog file...')
            pdtime_limits.append(float('nan'))
            pdtime_queue.append(float('nan'))
            pdtime_run.append(float('nan'))

    pdtime_limits=impute_nan_mean(pdtime_limits)
    pdtime_queue=impute_nan_mean(pdtime_queue)
    pdtime_run=impute_nan_mean(pdtime_run)
    pdtuples=list(zip(pdtime_limits, pdnodes, pdtime_queue, pdtime_run))


    rnodes = []
    rtime_limits = []
    rtime_queue = []
    rtime_run = []
    for jobid, nnodes in rjobs:
        rnodes.append(nnodes)
        if (jobid in job_comp_log_dict):
            rtime_limits.append(job_comp_log_dict[jobid]['Timelimit'])

            queuetime = job_comp_log_dict[jobid]['Start'] - job_comp_log_dict[jobid]['Submit']
            queuetime_numeric = max(0.0, queuetime.total_seconds() / (60 * 60))
            rtime_queue.append(queuetime_numeric)

            runtime = timestamp - job_comp_log_dict[jobid]['Start']
            runtime_numeric = max(0.0, runtime.total_seconds() / (60 * 60))
            rtime_run.append(runtime_numeric)

        else:
            if (debug):
                print(f'Warning: {timestamp} - running Job {jobid} is not included in jobcomplog file...')
            rtime_limits.append(float('nan'))
            rtime_queue.append(float('nan'))
            rtime_run.append(float('nan'))

    rtime_limits = impute_nan_mean(rtime_limits)
    rtime_queue = impute_nan_mean(rtime_queue)
    rtime_run = impute_nan_mean(rtime_run)
    rtuples = list(zip(rtime_limits, rnodes, rtime_queue, rtime_run))


    return [
                timestamp,
                pdtuples,
                rtuples
            ]


def construct_fake_interruption_overlap_sample(jobid, comp_dict, squeue_dict, debug=True):

    if(jobid in comp_dict):
        timestamp_submit=comp_dict[jobid]['Submit']

        if(timestamp_submit not in squeue_dict):
            if(debug):
                print(f'Warning: Job {jobid} SUBMIT timestamp {timestamp_submit} is not included in squeue logs...')
            return None
        else:
            time_window_nsteps=0
            time_window_snapshots=[]
            time_window_num_imputation=0
            while(len(time_window_snapshots)<TIME_WINDOW_NUM_STEPS):
                timestamp_lookup=timestamp_submit-timedelta(hours=TIME_WINDOW_STEP_SIZE*time_window_nsteps)
                if(timestamp_lookup in squeue_dict):
                    time_window_snapshots.append(list(squeue_dict[timestamp_lookup]))
                else:
                    if(debug):
                        print(f'Warning: Cannot find timestamp {timestamp_lookup} in squeue logs, skip...')
                    time_window_num_imputation+=1
                    if(time_window_num_imputation>TIME_WINDOW_IMPUTATION):
                        break
                time_window_nsteps+=1

            if(len(time_window_snapshots)==TIME_WINDOW_NUM_STEPS and time_window_num_imputation<=TIME_WINDOW_IMPUTATION):

                completion_time=comp_dict[jobid]['CompletionTime']

                if(np.random.uniform(0,1)<0.5):
                    rnd_running=0
                else:
                    rnd_running=1
                rnd=np.random.uniform(0,1)
                pred_fake_run_time=completion_time*rnd_running*rnd

                pred_fake_queue_time=np.random.uniform(0,3)

                succ_fake_queue_time= comp_dict[jobid]['QueueingDelay']
                succ_fake_run_time=0.0

                fake_interruption_overlap=succ_fake_queue_time-(completion_time-pred_fake_run_time)
                fake_nnodes = comp_dict[jobid]['NNodes']
                fake_timelimit=comp_dict[jobid]['Timelimit']

                raw_features=[]
                for snapshot in time_window_snapshots:
                    snapshot_fake=snapshot+[(fake_timelimit, fake_nnodes, pred_fake_queue_time, pred_fake_run_time)]
                    snapshot_fake+=[(fake_timelimit, fake_nnodes, succ_fake_queue_time, succ_fake_run_time)]
                    raw_features.append(snapshot_fake)

                return (raw_features, fake_interruption_overlap)
            else:
                return None

    else:
        if (debug):
            print(f'Warning: Job {jobid} is not included in jobcomp logs...')
        return None



if __name__ == '__main__':

    ray.init()


    squeue_log=squeue_log_load(SQUEUE_LOG_PATH, SQUEUE_LOG_SEEK_START,
                               SQUEUE_LOG_SEEK_END, SQUEUE_LOG_VALID_RECORD_MIN_NLINES)

    squeue_log_partitions=partition_list(squeue_log, SQUEUE_LOG_PARTITION_SIZE)
    squeue_log_partitions=[squeue_partition_parser.remote(partition) for partition in squeue_log_partitions]

    squeue_log_collection=coalesce_lists(ray.get(squeue_log_partitions))
    squeue_log_timestamps=[record[0] for record in squeue_log_collection]
    squeue_log_time_min=min(squeue_log_timestamps)
    squeue_log_time_max=max(squeue_log_timestamps)
    print(f'Time range with padding: [{squeue_log_time_min},{squeue_log_time_max}]')

    squeue_log_time_min_selected= squeue_log_time_min + TIME_RANGE_PADDING
    squeue_log_time_max_selected= squeue_log_time_max - TIME_RANGE_PADDING
    assert(squeue_log_time_max_selected > squeue_log_time_min_selected)
    print(f'Selected time range with: [{squeue_log_time_min_selected},{squeue_log_time_max_selected}]')



    jobcomp_log_dict=jobcomp_log_load(JOB_COMPLETION_LOG_PATH)
    jobcomp_log_dict_in_time_range={}
    for k,v in jobcomp_log_dict.items():
        if(v['Submit']>=squeue_log_time_min_selected and v['Submit']<=squeue_log_time_max_selected):
            jobcomp_log_dict_in_time_range[k]=v

    print(f'Number of jobs in jobcomp.log: {len(jobcomp_log_dict)}')
    print(f'Number of jobs in selected time range with padding: {len(jobcomp_log_dict_in_time_range)}')

    print(f'Generating squeue features...')
    jobcomp_dict_ds = ray.put(jobcomp_log_dict)
    squeue_partition_features=[squeue_partition_extract_features.remote(partition, jobcomp_dict_ds) for partition in squeue_log_partitions]
    squeue_feature_collection = coalesce_lists(ray.get(squeue_partition_features))


    '''
    Squeue feature format: [Timestamp,
                             
                            [...(PD_Time_limit, PD_NNodes, PD_Age)...], 
                            [...(R_Time_limit, R_NNodes, R_Runtime)...])
    '''
    squeue_feature_dict = {timestamp: [pdtuples, rtuples] for timestamp, pdtuples, rtuples in squeue_feature_collection}
    training_samples = []
    for jobid in jobcomp_log_dict_in_time_range.keys():
        print(f'Retrieving time-window snapshots (features) and interruption-overlap for Job {jobid}...')
        sample = construct_fake_interruption_overlap_sample(jobid, jobcomp_log_dict_in_time_range, squeue_feature_dict)
        training_samples.append(sample)
    print(f'{len(training_samples)} samples are constructed from squeue and jobcom log files in selected time range.')
    with open('data/fake_batch.pickle', 'wb') as output:
        pickle.dump(training_samples, output)




