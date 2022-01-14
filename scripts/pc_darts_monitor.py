import curses
import datetime
import os

import subprocess

SECONDS_TO_EXPIRE = 30
TIME_BETWEEN_UPDATES = 300
HOME_DIR = '/home/rafael'
# HOME_DIR = '/home/lovelace/proj/proj856/rafacsan'
MOUNT_DIR = '{}/workspace/mount'.format(HOME_DIR)
WIN_W = 80
WIN_H = 40


def list_worker_dirs():
    all_dirs = os.listdir(MOUNT_DIR)
    return list(filter(lambda x: 'adagp' in x, all_dirs))


def get_exp_ids(worker_dir):
    all_dirs = os.listdir('{}/{}/PC-DARTS'.format(MOUNT_DIR, worker_dir))
    return list(filter(lambda x: '-EXP-' in x, all_dirs))


def extract_t_from_line(line):
    splitted = line.split()
    date_str = ' '.join(splitted[:2])
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S,%f')


def check_if_running(log_path):
    line = subprocess.check_output(['tail', '-1', log_path])
    last = extract_t_from_line(line)
    current = datetime.datetime.now()
    return (current - last).total_seconds() < SECONDS_TO_EXPIRE


def get_time_per_epoch(p):
    times = []
    with open(p, 'r') as fp:
        last = None
        current = None
        for line in fp:
            if 'epoch ' in line:
                current = extract_t_from_line(line)
                if last is not None:
                    times.append((current - last).total_seconds())
                last = current
    return sum(times) / len(times) if len(times) > 0 else 999999


def get_eta(p):
    with open(p, 'r') as fp:
        last_epoch, epochs, _ = parse_file(fp)
    remaining_s = (epochs - last_epoch) * get_time_per_epoch(p)
    return datetime.datetime.now() + datetime.timedelta(seconds=remaining_s)


def get_log_file_path_and_status(worker_dir, exp_id):
    p = '{}/{}/PC-DARTS/{}/log.txt'.format(MOUNT_DIR, worker_dir, exp_id)
    running = check_if_running(p)
    return p, running, get_eta(p) if running else None


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


def get_exp_logs():
    logs = {}
    for worker_dir in list_worker_dirs():
        for exp_id in get_exp_ids(worker_dir):
            logs[exp_id] = get_log_file_path_and_status(worker_dir, exp_id)
    return logs


def update_exp_data(exp_data, exp_logs):
    for e_id, (path, running, eta) in exp_logs.items():
        if e_id not in exp_data or running:
            with open(path, 'r') as fp:
                last_epoch, epochs, validated_arch = parse_file(fp)
                exp_data[e_id] = (last_epoch, epochs, validated_arch)


def make_char_field(size, text, h_cursor):
    return text[:min(len(text), size)].ljust(size) + ' ', h_cursor + size + 1


def make_progress_bar(last_epoch, epochs, h_cursor):
    last_epoch = max(last_epoch, 0)
    bar_size = WIN_W - h_cursor - 3
    filled_bar_size = (bar_size * last_epoch) // (epochs - 1)
    numbers = '{}/{}'.format(last_epoch, epochs - 1)
    n_pos = (bar_size // 2) - len(numbers) // 2
    text = ['|']
    text.extend(['#' for _ in range(filled_bar_size)])
    text.extend(['-' for _ in range(bar_size - filled_bar_size)])
    text.append('|')
    for _ in range(len(numbers)):
        text.pop(n_pos)
    text.insert(n_pos, numbers)
    return ''.join(text), h_cursor + len(text)


def add_fields(main_pad, e_id, validated_arch, last_epoch, epochs, status, selected):
    h_cursor = 0
    id_field, h_cursor = make_char_field(28, e_id, h_cursor)
    arch_field, h_cursor = make_char_field(3, validated_arch if validated_arch else '', h_cursor)
    status_field, h_cursor = make_char_field(7, status, h_cursor)
    progress_bar, h_cursor = make_progress_bar(last_epoch, epochs, h_cursor)
    main_pad.addstr(
        '{}{}{}{}\n'.format(id_field, arch_field, status_field, progress_bar),
        curses.A_REVERSE if selected else curses.A_NORMAL)


def add_expanded_fields(main_pad, exp_logs, e_id):
    main_pad.addstr('\tEXP_DIR: {}\n'.format(exp_logs[e_id][0]))
    if exp_logs[e_id][1]:
        date_str = exp_logs[e_id][2].strftime('%Y-%m-%d %H:%M:%S,%f')
        main_pad.addstr('\tETA: {}\n'.format(date_str))


def show_exp_data(exp_logs, exp_data, main_pad, cursor, expanded, expand_toggle):
    i = 0
    main_pad.clear()
    for e_id, (last_epoch, epochs, validated_arch) in exp_data.items():
        if expand_toggle and i == cursor:
            expanded[e_id] = not expanded[e_id]
        status = 'RUNNING' if exp_logs[e_id][1] else 'IDLE'
        add_fields(main_pad, e_id, validated_arch, last_epoch, epochs, status, cursor == i)
        if expanded[e_id]:
            add_expanded_fields(main_pad, exp_logs, e_id)
        i += 1


def update_cursor(input_char, cursor, top):
    new_cursor = cursor
    expand = False
    if input_char and input_char == curses.KEY_UP:
        new_cursor = cursor - 1
    elif input_char and input_char == curses.KEY_DOWN:
        new_cursor = cursor + 1
    elif input_char and input_char == ord(' '):
        expand = True
    if new_cursor != cursor:
        top = update_top(new_cursor, top)
    return new_cursor, top, expand


def update_top(cursor, top):
    if cursor - top == WIN_H:
        return top + 1
    elif cursor - top < 0:
        return top - 1
    else:
        return top


def begin_monitor(std_scr):
    global WIN_H, WIN_W
    last_update = datetime.datetime.now()
    exp_logs = get_exp_logs()
    exp_data = {}
    update_exp_data(exp_data, exp_logs)
    cursor = 0
    top = 0
    expanded = {e_id: False for e_id in exp_logs}
    expand_toggle = False
    main_pad = curses.newpad(200, WIN_W)
    while True:
        WIN_H, WIN_W = std_scr.getmaxyx()
        main_pad.resize(200, WIN_W)
        if (datetime.datetime.now() - last_update).total_seconds() > TIME_BETWEEN_UPDATES:
            exp_logs = get_exp_logs()
            update_exp_data(exp_data, exp_logs)
            expanded = {e_id: False if e_id not in expanded else expanded[e_id] for e_id in exp_logs}
            last_update = datetime.datetime.now()
        show_exp_data(exp_logs, exp_data, main_pad, cursor, expanded, expand_toggle)
        input_char = std_scr.getch()
        if input_char == ord('q'):
            break
        elif input_char is None:
            expand_toggle = False
        else:
            cursor, top, expand_toggle = update_cursor(input_char, cursor, top)
        # main_pad.addstr('{}'.format(exp_data))
        try:
            main_pad.refresh(top, 0, 0, 0, WIN_H - 1, WIN_W - 1)
        except curses.error:
            continue
        curses.delay_output(100)


def main(std_scr):
    global WIN_H, WIN_W
    WIN_H, WIN_W = std_scr.getmaxyx()
    std_scr.nodelay(True)
    begin_monitor(std_scr)
    std_scr.nodelay(False)


if __name__ == '__main__':
    curses.wrapper(main)
