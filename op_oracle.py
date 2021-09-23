import torch
import torch.nn.functional as F
import logging
from torch import nn

from genotypes import PRIMITIVES

OPERATION_LOSS_W = 1.0


class OpPerformanceOracle:
    def __init__(self):
        self.weights = {}
        self.softmaxed_weights = None

    def reset_weights(self):
        self.weights.clear()
        for p in PRIMITIVES:
            self.weights[p] = None

    def set_default_weights(self):
        self.reset_weights()
        self.set_weight('none', 2)
        self.set_weight('max_pool_3x3', 2)
        self.set_weight('avg_pool_3x3', 2)
        self.set_weight('skip_connect', 2)
        self.set_weight('sep_conv_3x3', 1)
        self.set_weight('sep_conv_5x5', 2)
        self.set_weight('dil_conv_3x3', 2)
        self.set_weight('dil_conv_5x5', 2)
        self._compute_softmaxed_weights()

    def set_weight(self, operation, weight):
        self.weights[operation] = weight

    def _weights_are_valid(self):
        for weight in self.weights.values():
            if not weight:
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
            a_softmax = F.softmax(alpha)
            for i in range(alpha.dim()):
                weighted_alphas.append(self.weights[PRIMITIVES[i]] * a_softmax[i])
        return sum(weighted_alphas) / (total_weight * len(network_cells_alphas))

    def get_operation_rate_v2(self, network_cells_alphas):
        softmaxes_diff = []
        if not self._weights_are_valid():
            raise RuntimeError('Undefined weights')
        for alpha in network_cells_alphas:
            if len(self.weights) != len(alpha):
                raise RuntimeError('Incorrect dimension for Alpha')
            a_softmax = F.softmax(alpha)
            softmaxes_diff.append(torch.sum(torch.abs(a_softmax - self.softmaxed_weights)))
        return sum(softmaxes_diff) / len(network_cells_alphas)

    def _compute_softmaxed_weights(self):
        weight_array = []
        for op in PRIMITIVES:
            weight_array.append(self.weights[op])
        weight_tensor = torch.cuda.FloatTensor(weight_array)
        self.softmaxed_weights = F.softmax(torch.autograd.Variable(weight_tensor), dim=0)


class CustomLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, current_alpha=None, oracle=None):
        super(CustomLoss, self).__init__(weight, size_average)
        self.current_network_cells_alphas = current_alpha
        self.oracle = oracle

    def update_current_network_cells_alphas(self, network_cells_alphas):
        self.current_network_cells_alphas = network_cells_alphas

    def forward(self, input, target):
        cross_entropy_loss = super(CustomLoss, self).forward(input, target)
        op_rate = self.oracle.get_operation_rate_v2(self.current_network_cells_alphas) if self.oracle else 0.0
        op_loss = op_rate * OPERATION_LOSS_W
        logging.info('LOSS = {} + {}'.format(cross_entropy_loss.data[0], op_loss.data[0]))
        return cross_entropy_loss + op_loss
