#!/usr/bin/env python
import datetime
import tarfile

from scripts.arch_data import ArchDataCollection

EXP_DIR = '/home/rafael/Projetos/msc-rafael-cortez-sanchez/labbook/results'


def extract_t_from_line(line):
    splitted = line.split()
    date_str = ' '.join(splitted[:2])
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')


def get_elapsed_time(fp):
    start_t = None
    end_t = None
    for line in fp:
        try:
            current_t = extract_t_from_line(line)
        except ValueError:
            continue
        start_t = current_t if start_t is None or current_t < start_t else start_t
        end_t = current_t if end_t is None or current_t > end_t else end_t
    return (end_t - start_t).total_seconds() / (60 * 60)


def extract_elapsed_time(train_id):
    if not train_id:
        return None
    ts_f = tarfile.open('{}/{}.tar.gz'.format(EXP_DIR, train_id))
    ts_f.extractall('/tmp/{}'.format(train_id))
    with open('/tmp/{}/log.txt'.format(train_id), 'r') as fp:
        elapsed_time = get_elapsed_time(fp)
    return elapsed_time


def main():
    arch_data_collection = ArchDataCollection()
    arch_data_collection.load()
    archs = arch_data_collection.archs
    for a_id in archs:
        ts_id = archs[a_id].train_search_id
        t_id = archs[a_id].best_train_id
        elapsed_time_ts = extract_elapsed_time(ts_id)
        elapsed_time_t = extract_elapsed_time(t_id)
        print(a_id, elapsed_time_ts, elapsed_time_t)


if __name__ == '__main__':
    main()
