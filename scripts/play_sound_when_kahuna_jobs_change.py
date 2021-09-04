#!/usr/bin/python3
import logging
import subprocess
import time

SLEEP_TIME = 60
logging.basicConfig(filename='/home/rafael/.kahuna_watch', format='%(asctime)s %(message)s', level=logging.INFO)


def play_siren():
    try:
        subprocess.run(['aplay', '/home/rafael/alarm.wav'])
    except subprocess.CalledProcessError:
        logging.error('Failed to play siren')


def fetch_job_list():
    try:
        output = subprocess.run(['ssh', 'rsanchez@kahuna.iqm.unicamp.br', 'qstat', '-u', 'rsanchez'],
                                stdout=subprocess.PIPE)
        output = output.stdout.decode('utf-8')
    except subprocess.CalledProcessError:
        logging.error('Failed to fetch jobs from Kahuna')
        return None
    output = output.split('\n')
    i = 0
    while True:
        if i >= len(output):
            return None
        i += 1
        if 'Job ID' in output[i - 1]:
            i += 1  # --- sep line
            break
    jobs = [line.split() for line in output[i:]]
    return list(filter(lambda j: len(j) > 0, jobs))


def jobs_changed(current_list, previous_list):
    cl_ids = [job[0] for job in current_list]
    pl_ids = [job[0] for job in previous_list]
    changed = False
    for j_id in pl_ids:
        if j_id not in cl_ids:
            logging.info('Job {} finished'.format(j_id))
            changed = True
    for j_id in cl_ids:
        if j_id not in pl_ids:
            logging.info('Job {} started'.format(j_id))
            changed = True
    return changed


def log_job_info(current_jobs):
    logging.info('Current jobs:')
    logging.info(
        ' '.join(['JobID', 'Username', 'Queue', 'Jobname', 'SessID', 'NDS', 'TSK Memory', 'Time', 'STime']))
    for j in current_jobs:
        logging.info(' '.join(j))


def main():
    current_jobs = None
    logging.info('Fetching first job list...')
    while not current_jobs:
        current_jobs = fetch_job_list()
        time.sleep(SLEEP_TIME)

    notification_counter = 0
    logging.info('Begin monitoring Kahuna...')
    while True:
        if notification_counter % 30 == 0:
            log_job_info(current_jobs)
        fetched_jobs = fetch_job_list()
        if fetched_jobs and jobs_changed(current_jobs, fetched_jobs):
            current_jobs = fetched_jobs
            play_siren()
            log_job_info(current_jobs)
        time.sleep(SLEEP_TIME)
        notification_counter += 1


if __name__ == '__main__':
    main()
