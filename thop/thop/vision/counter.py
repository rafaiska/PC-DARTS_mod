import torch
import numpy as np


def counter_parameters(para_list):
    total_params = 0
    for p in para_list:
        total_params += torch.FloatTensor([p.nelement()])
    return total_params


def counter_zero_ops():
    return torch.FloatTensor([int(0)])


def counter_conv(bias, kernel_size, output_size, in_channel, group):
    """inputs are all numbers!"""
    return torch.FloatTensor([output_size * (in_channel / group * kernel_size + bias)])


def counter_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.FloatTensor([2 * input_size])


def counter_relu(input_size):
    return torch.FloatTensor([int(input_size)])


def counter_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.FloatTensor([int(total_ops)])


def counter_avgpool(input_size):
    return torch.FloatTensor([int(input_size)])


def counter_adap_avg(kernel_size, output_size):
    total_div = 1
    kernel_op = kernel_size + total_div
    return torch.FloatTensor([int(kernel_op * output_size)])


def counter_upsample(mode, output_size):
    total_ops = output_size
    if mode == "linear":
        total_ops *= 5
    elif mode == "bilinear":
        total_ops *= 11
    elif mode == "bicubic":
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops *= (ops_solve_A + ops_solve_p)
    elif mode == "trilinear":
        total_ops *= (13 * 2 + 5)
    return torch.FloatTensor([int(total_ops)])


def counter_linear(in_feature, num_elements):
    return torch.FloatTensor([int(in_feature * num_elements)])


def counter_matmul(input_size, output_size):
    input_size = np.array(input_size)
    output_size = np.array(output_size)
    return np.prod(input_size) * output_size[-1]


def counter_mul(input_size):
    return input_size


def counter_pow(input_size):
    return input_size


def counter_sqrt(input_size):
    return input_size


def counter_div(input_size):
    return input_size
