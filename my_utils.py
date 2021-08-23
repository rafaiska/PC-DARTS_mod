import ast
import os

import genotypes
from utils import load

SUCCEEDED_EXP_STR = 'valid_acc'
CHECKPOINT_FILE = 'checkpoint.info'


def _parse_list_in_line(line):
    first_bracket_pos = -1
    for _ in range(line.count('[')):
        first_bracket_pos = line.find('[', first_bracket_pos + 1)
    parsed_str = line[first_bracket_pos:line.find(']') + 1]
    return ast.literal_eval(parsed_str)


def _add_to_exp_data(current_list, exp_data, node_index):
    node_data = exp_data[node_index]
    for i in range(len(genotypes.PRIMITIVES)):
        node_data[genotypes.PRIMITIVES[i]].append(current_list[i])


def _extract_from_lines(file_pt, first_line, exp_data):
    current_list = _parse_list_in_line(first_line)
    _add_to_exp_data(current_list, exp_data, 0)
    node_index = 1
    while True:
        current_line = file_pt.readline()
        current_list = _parse_list_in_line(current_line)
        _add_to_exp_data(current_list, exp_data, node_index)
        node_index += 1
        if current_line.count(']') >= 2:
            break


def extract_from_exp_data(dir_path):
    exp_data_normal = [{op: [] for op in genotypes.PRIMITIVES} for _ in range(14)]
    exp_data_reduce = [{op: [] for op in genotypes.PRIMITIVES} for _ in range(14)]
    file_pt = open(dir_path + '/log.txt', 'r')
    while True:
        line = file_pt.readline()
        if not line:
            break
        if 'ALPHA' in line:
            if 'NORMAL' in line:
                _extract_from_lines(file_pt, line, exp_data_normal)
            elif 'REDUCE' in line:
                _extract_from_lines(file_pt, line, exp_data_reduce)
            else:
                raise RuntimeError('Unexpected line')
    file_pt.close()
    return exp_data_normal, exp_data_reduce


def _check_if_succeeded(lines):
    last_valid_line = lines[-1] if len(lines) >= 1 else ''
    return SUCCEEDED_EXP_STR in last_valid_line


def _get_epochs(lines):
    for i in range(len(lines) - 1, -1, -1):
        if 'epoch ' in lines[i]:
            splitted = lines[i].split()
            epoch = splitted[splitted.index('epoch') + 1]
            return int(epoch)
    return None


def _extract_exp_info(dir):
    with open(dir + '/log.txt', 'r') as fp:
        lines = fp.readlines()
    return _get_epochs(lines), _check_if_succeeded(lines)


def list_experiment_status(search_exp_dir):
    output = open('exp_results.csv', 'w')
    for dir_name in sorted(os.listdir(search_exp_dir)):
        if 'search-EXP-' in dir_name:
            epochs, succeeded = _extract_exp_info('/'.join([search_exp_dir, dir_name]))
            output.write(','.join([dir_name, 'T' if succeeded else 'F', str(epochs)]) + '\n')


def update_checkpoint(exp_dir, current_epoch):
    with open(CHECKPOINT_FILE, 'w') as fp:
        fp.write('{}\n'.format(exp_dir))
        fp.write(str(current_epoch))


def get_checkpoint_info():
    with open(CHECKPOINT_FILE, 'r') as fp:
        exp_dir = fp.readline()
        current_epoch = int(fp.readline())
    return exp_dir, current_epoch


def load_checkpoint_model(model, exp_dir):
    model_path = '{}/weights.pt'.format(exp_dir)
    load(model, model_path)


def clear_checkpoint():
    os.remove(CHECKPOINT_FILE)


def print_cell_edges_from_alphas(alphas):
    i = 0
    for alpha_o in alphas:
        max_index = alpha_o.index(max(alpha_o))
        op_name = genotypes.PRIMITIVES[max_index]
        print('Edge {}: {}'.format(i, op_name))
        i += 1
