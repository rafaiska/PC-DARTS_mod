#!/usr/bin/python3
import subprocess
import time
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

TIME_MS_BETWEEN_CHECKS = 30000
TIME_MS_FOR_STARTUP = 10000
SSH_TIMEOUT_S = 6
USERNAME = 'ra094324'
SSH_COMMAND = ['ssh', '-o', 'ConnectTimeout={}'.format(SSH_TIMEOUT_S), '{}@lmcad-dl-1.ic.unicamp.br'.format(USERNAME)]
RUN_COMMAND = ['./run_with_nohup.sh']
CHECK_PROCESS_COMMAND = ['ps', 'aux', '|', 'grep', 'docker']
PC_DARTS_DIR = '~/workspace/PC-DARTS'
CHECK_DIRS_COMMAND = ['ls', PC_DARTS_DIR]
GREP_PROCESS_STR = 'grep'
SUCCEEDED_EXP_STR = 'valid_acc'


def run_command(command):
    logging.info('Calling command: \"${}\"...'.format(' '.join(command)))
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None
    logging.info('Command returned {}'.format(result.returncode))
    return result.stdout.decode('utf-8')


def run_main_command_without_out(command):
    logging.info('Calling command without output: \"${}\"...'.format(' '.join(command)))
    try:
        subprocess.Popen(command)
    except subprocess.CalledProcessError:
        return


def check_if_succeeded(run_id):
    if not run_id:
        return False
    experiment_log_filename = PC_DARTS_DIR + '/{}/log.txt'.format(run_id)
    cat_command = []
    cat_command.extend(SSH_COMMAND)
    cat_command.extend(['cat', experiment_log_filename])
    output = run_command(cat_command).split('\n')
    if not output:
        raise RuntimeError('Attempt to cat nonexistent log file: {}'.format(experiment_log_filename))
    last_valid_line = output[-2] if len(output) >= 2 else ''
    return SUCCEEDED_EXP_STR in last_valid_line


def check_if_running():
    command = []
    command.extend(SSH_COMMAND)
    command.extend(CHECK_PROCESS_COMMAND)
    output = run_command(command)
    if not output:
        return True  # FIXME: WARNING: It may not be running even if server fails to provide output
    for line in output.split('\n'):
        splitted = line.split()
        if len(splitted) > 0 and splitted[0] == USERNAME and GREP_PROCESS_STR not in line:
            return True
    return False


def get_last_search_exp(c_output):
    exp_list = filter(lambda s: 'search-EXP-' in s, c_output)
    return sorted(exp_list)[-1]


def retrieve_exp_id():
    command = []
    command.extend(SSH_COMMAND)
    command.extend(CHECK_DIRS_COMMAND)
    output = run_command(command)
    return get_last_search_exp(output.split('\n')) if output else None


def wait_ms(time_ms):
    time.sleep(time_ms / 1000)


def make_run():
    command = []
    command.extend(SSH_COMMAND)
    command.extend(RUN_COMMAND)
    run_main_command_without_out(command)
    wait_ms(TIME_MS_FOR_STARTUP)
    return retrieve_exp_id() if check_if_running() else None


def main():
    succeeded = False
    current_run_id = None
    while not succeeded:
        if not check_if_running() and not check_if_succeeded(current_run_id):
            logging.info('Experiment {} not running. Starting a new run...'.format(current_run_id))
            current_run_id = make_run()
            logging.info('Run ID is now {}'.format(current_run_id))
        else:
            logging.info('Still running {}...'.format(current_run_id if current_run_id else 'previous experiment'))
        wait_ms(TIME_MS_BETWEEN_CHECKS)
        succeeded = check_if_succeeded(current_run_id)
    logging.info('Experiment {} seems to have succeeded.'.format(current_run_id))


if __name__ == '__main__':
    main()
