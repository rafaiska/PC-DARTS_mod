import logging
import sys

import torch
import torch.nn.functional as F
from torch import nn

import thop
from genotypes import PRIMITIVES
from model import NetworkCIFAR
from operations import OPS

REPLACE_ZERO_RATE = 0.8
REDUCE_IMPORTANCE = 2.0 / 20.0
MAX_OP_LOSS = torch.autograd.Variable(torch.cuda.FloatTensor([3.0]))
PYTHON_3 = sys.version[0] == '3'


def get_n_inputs():
    n_inputs = [torch.randn(1, 3, 2, 2).cuda(),
                torch.randn(1, 3, 4, 4).cuda(),
                torch.randn(1, 3, 8, 8).cuda(),
                torch.randn(1, 3, 16, 16).cuda(),
                torch.randn(1, 3, 32, 32).cuda(),
                torch.randn(1, 3, 64, 64).cuda(),
                torch.randn(1, 3, 128, 128).cuda(),
                ]
    return n_inputs


class FPOpCounter:
    """
    Based on https://github.com/clavichord93/CNN-Calculator
    https://arxiv.org/abs/1811.03060
    """

    def __init__(self, use_thop=False):
        self.layers = None
        self.genotype = None
        self.min_fp_op = None
        self.max_fp_op = None
        self.last_fp_op = None
        self.use_thop = use_thop

    def setup(self, input_width, input_height, n_layers, init_channels):
        self.layers = []
        c_curr, c_prev = init_channels, init_channels
        input_s_curr = input_s_prev = (input_width, input_height)
        for i in range(n_layers):
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_curr *= 2
                input_s_curr = tuple(int(x / 2) for x in input_s_curr)
            cell = (c_prev, c_curr, input_s_prev, input_s_curr)
            c_prev = c_curr
            input_s_prev = input_s_curr
            self.layers.append(cell)

    @staticmethod
    def conv2d(w_in, h_in, c_in, c_out, kernel_size, stride, padding, dilation, groups, bias):
        eff_ks = kernel_size + (kernel_size - 1) * (dilation - 1)
        out_h = (h_in - eff_ks + 2 * padding) // stride + 1
        out_w = (w_in - eff_ks + 2 * padding) // stride + 1

        # fp_ops = (c_out * c_in // groups * kernel_size * kernel_size + 1) * out_h * out_w
        fp_ops = (c_in * c_out // groups * kernel_size * kernel_size) * out_h * out_w
        if bias:
            fp_ops += c_out * out_w * out_h

        return fp_ops

    @staticmethod
    def pool2d(w_in, h_in, c, kernel_size, stride, padding):
        out_h = (h_in - kernel_size + 2 * padding) // stride + 1
        out_w = (w_in - kernel_size + 2 * padding) // stride + 1
        # return c * out_h * out_w * kernel_size * kernel_size
        return c * out_h * out_w

    @staticmethod
    def _identity():
        return 0

    @staticmethod
    def _batch_norm():
        return 0

    @staticmethod
    def none():
        return 0

    @staticmethod
    def avg_pool_3x3(w_in, h_in, c, stride):
        return FPOpCounter.pool2d(w_in, h_in, c, 3, stride, 1)

    @staticmethod
    def max_pool_3x3(w_in, h_in, c, stride):
        # return FPOpCounter.pool2d(w_in, h_in, c, 3, stride, 1)
        return 0

    @staticmethod
    def _factorized_reduce(w_in, h_in, c):
        ops = [FPOpCounter.conv2d(w_in, h_in, c, c // 2, 1, 2, 0, 1, 1, False),
               FPOpCounter.conv2d(w_in, h_in, c, c // 2, 1, 2, 0, 1, 1, False),
               FPOpCounter._batch_norm()]
        return sum(ops)

    @staticmethod
    def skip_connect(w_in, h_in, c, stride):
        return FPOpCounter._identity() if stride == 1 else FPOpCounter._factorized_reduce(w_in, h_in, c)

    @staticmethod
    def _sep_conv(w_in, h_in, c, kernel_size, stride, padding):
        ops = [FPOpCounter.conv2d(w_in, h_in, c, c, kernel_size, stride, padding, 1, c, False),
               FPOpCounter.conv2d(w_in, h_in, c, c, 1, 1, 0, 1, 1, False),
               FPOpCounter._batch_norm(),
               FPOpCounter.conv2d(w_in, h_in, c, c, kernel_size, 1, padding, 1, c, False),
               FPOpCounter.conv2d(w_in, h_in, c, c, 1, 1, 0, 1, 1, False),
               FPOpCounter._batch_norm(),
               ]
        return sum(ops)

    @staticmethod
    def sep_conv_3x3(w_in, h_in, c, stride):
        return FPOpCounter._sep_conv(w_in, h_in, c, 3, stride, 1)

    @staticmethod
    def sep_conv_5x5(w_in, h_in, c, stride):
        return FPOpCounter._sep_conv(w_in, h_in, c, 5, stride, 2)

    @staticmethod
    def sep_conv_7x7(w_in, h_in, c, stride):
        return FPOpCounter._sep_conv(w_in, h_in, c, 7, stride, 3)

    @staticmethod
    def _dil_conv(w_in, h_in, c, kernel_size, stride, padding):
        ops = [FPOpCounter.conv2d(w_in, h_in, c, c, kernel_size, stride, padding, 2, c, False),
               FPOpCounter.conv2d(w_in, h_in, c, c, 1, stride, 0, 1, 1, False),
               FPOpCounter._batch_norm()]
        return sum(ops)

    @staticmethod
    def dil_conv_3x3(w_in, h_in, c, stride):
        return FPOpCounter._dil_conv(w_in, h_in, c, 3, stride, 2)

    @staticmethod
    def dil_conv_5x5(w_in, h_in, c, stride):
        return FPOpCounter._dil_conv(w_in, h_in, c, 5, stride, 4)

    def update_genotype_from_network(self, network):
        self.genotype = network.genotype()
        if not self.use_thop:
            fp_ops = self.count_network_fp_ops()
        else:
            validation_net = NetworkCIFAR(36, 10, 20, False, self.genotype).cuda()
            validation_net.drop_path_prob = 0.0
            inpt = torch.autograd.Variable(torch.randn(1, 3, 32, 32).cuda())
            fp_ops, _ = thop.profile(validation_net, inputs=(inpt,), verbose=False)

        self.min_fp_op = fp_ops if not self.min_fp_op or fp_ops < self.min_fp_op else self.min_fp_op
        self.max_fp_op = fp_ops if not self.max_fp_op or fp_ops > self.max_fp_op else self.max_fp_op
        self.last_fp_op = fp_ops

    @staticmethod
    def get_macs_from_model(network):
        validation_net = NetworkCIFAR(36, 10, 20, False, network.genotype()).cuda()
        validation_net.drop_path_prob = 0.0
        inpt = torch.autograd.Variable(torch.randn(1, 3, 32, 32).cuda())
        fp_ops, _ = thop.profile(validation_net, inputs=(inpt,), verbose=False)
        return fp_ops

    @staticmethod
    def _count_layer_op_fp_ops(op, src_node, reduce, output_c, input_s, output_s):
        op_counter = eval('FPOpCounter.{}'.format(op))
        stride = 2 if src_node < 2 and reduce else 1
        w_in, h_in = input_s if src_node < 2 else output_s
        return op_counter(w_in, h_in, output_c, stride)

    def _count_layer_fp_ops(self, layer):
        fp_ops = 0
        input_c, output_c, input_s, output_s = layer
        reduce = input_c != output_c
        sub_geno = self.genotype[2] if reduce else self.genotype[0]
        for dst_node in range(2, 6):
            for i in range(2):
                op, src_node = sub_geno[(dst_node - 2) * 2 + i]
                fp_ops += FPOpCounter._count_layer_op_fp_ops(op, src_node, reduce, output_c, input_s, output_s)
        return fp_ops

    def count_network_fp_ops(self):
        counter = 0
        for layer in self.layers:
            counter += self._count_layer_fp_ops(layer)
        return counter

    def get_current_fp_op_rate(self):
        """
        https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
        """
        numerator = self.last_fp_op - self.min_fp_op
        denominator = self.max_fp_op - self.min_fp_op
        # logging.info(
        #     'Current FP OP Rate: ({:.2f} - {:.2f}) / ({:.2f} - {:.2f})'.format(self.last_fp_op, self.min_fp_op,
        #                                                                        self.max_fp_op, self.min_fp_op))
        return numerator / denominator if denominator != 0 else 0


class OpPerformanceOracle:
    def __init__(self):
        self.weights = {}
        self.softmaxed_weights = None
        self.fp_op_counter = FPOpCounter(use_thop=True)

    def get_current_macs(self):
        return self.fp_op_counter.last_fp_op

    def setup_counter(self, input_w, input_h, n_layers, init_channels):
        self.fp_op_counter.setup(input_w, input_h, n_layers, init_channels)

    def update_genotype_from_network(self, network):
        self.fp_op_counter.update_genotype_from_network(network)

    def reset_weights(self):
        self.weights.clear()
        for p in PRIMITIVES:
            self.weights[p] = None

    def set_default_weights(self):
        self.reset_weights()
        self.set_weight('none', 1)
        self.set_weight('max_pool_3x3', 1)
        self.set_weight('avg_pool_3x3', 1)
        self.set_weight('skip_connect', 1)
        self.set_weight('sep_conv_3x3', 1)
        self.set_weight('sep_conv_5x5', 1)
        self.set_weight('dil_conv_3x3', 1)
        self.set_weight('dil_conv_5x5', 1)
        self._compute_softmaxed_weights()

    def _compute_average_macs(self, operation_name):
        op_constructor = OPS[operation_name]
        macs_list = []
        for i in get_n_inputs():
            op = op_constructor(i.shape[1], 1, True).cuda()
            macs, _ = thop.profile(op, inputs=(torch.autograd.Variable(i),), verbose=False)
            macs_list.append(macs)
        self.weights[operation_name] = sum(macs_list) / float(len(macs_list))

    def set_weights_from_macs(self):
        self.reset_weights()
        for p in PRIMITIVES:
            self._compute_average_macs(p)
        self._compute_softmaxed_weights()
        logging.info('COMPUTED ORACLE WEIGHTS: {}'.format(self.weights))

    def replace_zero_weights(self):
        """
        Adjust weights to avoid null weights on cheap, non convolutional operators
        :return: None
        """

        def _is_zero(n):
            return 0.1 > n >= 0.0

        def _get_min_not_zero():
            weights = list(self.weights.values())
            weights.sort(reverse=True)
            while _is_zero(weights[-1]):
                weights.pop()
            return weights[-1]

        if not self._weights_are_valid():
            raise RuntimeError
        min_nz = _get_min_not_zero()
        for op, w in self.weights.items():
            if _is_zero(w):
                self.weights[op] = min_nz * REPLACE_ZERO_RATE
        logging.info('ADJUSTED ORACLE WEIGHTS: {}'.format(self.weights))

    def set_weight(self, operation, weight):
        self.weights[operation] = weight

    def _weights_are_valid(self):
        for weight in self.weights.values():
            if weight is None:
                return False
        return True

    def get_operation_rate(self, network_cells_alphas):
        if not self._weights_are_valid():
            raise RuntimeError('Undefined weights')
        total_weight = sum(self.weights.values())
        weighted_alphas = []
        for alpha in network_cells_alphas:
            if len(self.weights) != len(alpha):
                raise RuntimeError('Incorrect dimension for Alpha')
            a_softmax = F.softmax(alpha, dim=0)
            for i in range(len(alpha)):
                weighted_alphas.append(self.weights[PRIMITIVES[i]] * a_softmax[i])
                # print('{}: {}*{}={}'.format(
                #     PRIMITIVES[i], self.weights[PRIMITIVES[i]], a_softmax[i], weighted_alphas[-1]))
        rate = float(sum(weighted_alphas)) / (total_weight * len(network_cells_alphas))
        return torch.autograd.Variable(torch.cuda.FloatTensor([rate]))

    def get_operation_rate_v2(self, network_cells_alphas):
        softmaxes_diff = []
        if not self._weights_are_valid():
            raise RuntimeError('Undefined weights')
        for alpha in network_cells_alphas:
            if len(self.weights) != len(alpha):
                raise RuntimeError('Incorrect dimension for Alpha')
            a_softmax = F.softmax(alpha, dim=0)
            softmaxes_diff.append(torch.sum(torch.abs(a_softmax - self.softmaxed_weights)))
        return sum(softmaxes_diff) / len(network_cells_alphas)

    def get_operation_rate_v3(self, network_cells_alphas):
        rate_tensor = torch.cuda.FloatTensor([self.fp_op_counter.get_current_fp_op_rate()])
        return torch.autograd.Variable(rate_tensor)

    def get_operation_rate_v4(self, alpha_normal, alpha_reduce):
        if not self._weights_are_valid():
            raise RuntimeError('Undefined weights')
        weights_tensor = torch.autograd.Variable(
            torch.cuda.FloatTensor([[self.weights[k] for k in PRIMITIVES] for _ in range(len(alpha_normal))]),
            requires_grad=True)
        normal_softmax = F.softmax(alpha_normal)
        reduce_softmax = F.softmax(alpha_reduce)
        normal_component = (weights_tensor * normal_softmax).sum()
        reduce_component = (weights_tensor * reduce_softmax).sum()
        return normal_component * (1 - REDUCE_IMPORTANCE) + reduce_component * REDUCE_IMPORTANCE

    def _compute_softmaxed_weights(self):
        weight_array = []
        for op in PRIMITIVES:
            weight_array.append(self.weights[op])
        weight_tensor = torch.cuda.FloatTensor(weight_array)
        self.softmaxed_weights = F.softmax(torch.autograd.Variable(weight_tensor), dim=0)


class CustomLoss(nn.CrossEntropyLoss):
    def __init__(self, oracle=None, closs_w=1 / 10e6):
        super(CustomLoss, self).__init__()
        self.alpha_normal = None
        self.alpha_reduce = None
        self.oracle = oracle
        self.c_loss_enabled = False
        self.c_loss_w = closs_w

    def enable_closs(self):
        self.c_loss_enabled = True

    def get_current_macs(self):
        return self.oracle.get_current_macs()

    def update_network_genotype_info(self, model):
        self.alpha_normal = model.alphas_normal
        self.alpha_reduce = model.alphas_reduce
        if self.oracle:
            self.oracle.update_genotype_from_network(model)

    def forward(self, input, target):
        cross_entropy_loss = super(CustomLoss, self).forward(input, target)
        if self.oracle and self.c_loss_enabled:
            op_rate = self.oracle.get_operation_rate_v4(self.alpha_normal, self.alpha_reduce)
        else:
            op_rate = torch.autograd.Variable(torch.cuda.FloatTensor([0.0]), requires_grad=True)
        op_loss = op_rate * self.c_loss_w
        final_loss = cross_entropy_loss + op_loss
        if PYTHON_3:
            logging.info('LOSS = {} + {} = {}'.format(
                cross_entropy_loss.data.item(), op_loss.data.item(), final_loss.data.item()))
        else:
            logging.info('LOSS = {} + {} = {}'.format(
                cross_entropy_loss.data[0], op_loss.data[0], final_loss.data[0]))
        return final_loss
