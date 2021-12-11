import os
import tarfile

HOME_DIR = '/home/lovelace/proj/proj856/rafacsan'
MOUNT_DIR = '{}/workspace/mount'.format(HOME_DIR)


def list_worker_dirs():
    all_dirs = os.listdir(MOUNT_DIR)
    return list(filter(lambda x: 'adagp' in x, all_dirs))


def get_exp_ids(worker_dir):
    all_dirs = os.listdir('{}/{}/PC-DARTS'.format(MOUNT_DIR, worker_dir))
    return list(filter(lambda x: '-EXP-' in x, all_dirs))


def get_log_file_path(worker_dir, exp_id):
    return '{}/{}/PC-DARTS/{}/log.txt'.format(MOUNT_DIR, worker_dir, exp_id)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname='.')


def parse_namespace(line):
    validated_arch = None
    epochs = None
    for s in line.split():
        if 'arch=' in s:
            ss = s.split('\'')
            validated_arch = ss[1]
        elif 'epochs=' in s:
            epochs = int(s[s.find('=') + 1:s.find(',')])
    return epochs, validated_arch


def parse_file(fp):
    epochs = None
    validated_arch = None
    last_epoch = -1
    for line in fp:
        if 'args = Namespace(' in line:
            epochs, validated_arch = parse_namespace(line)
        elif ' epoch ' in line:
            splitted = line.split()
            current_epoch = int(splitted[3])
            last_epoch = current_epoch if current_epoch > last_epoch else last_epoch
    return last_epoch, epochs, validated_arch


def tarfile_exists_for(exp_id):
    return os.path.isfile('{}/{}.tar.gz'.format(HOME_DIR, exp_id))


def find_finished_train_runs():
    for worker_dir in list_worker_dirs():
        for exp_id in get_exp_ids(worker_dir):
            if tarfile_exists_for(exp_id):
                print('tarfile exists for {}'.format('{}/PC-DARTS/{}'.format(worker_dir, exp_id)))
                continue
            log_file_p = get_log_file_path(worker_dir, exp_id)
            with open(log_file_p, 'r') as fp:
                last_epoch, epochs, validated_arch = parse_file(fp)
            completed = last_epoch == epochs - 1
            exp_dir = '{}/{}/PC-DARTS/{}'.format(MOUNT_DIR, worker_dir, exp_id)
            print(completed, validated_arch, exp_dir)
            if completed:
                make_tarfile('{}/{}.tar.gz'.format(HOME_DIR, exp_id), exp_dir)


if __name__ == '__main__':
    find_finished_train_runs()
