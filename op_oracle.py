import torch.nn.functional as F
from torch import nn

from genotypes import PRIMITIVES


class OpPerformanceOracle:
    def __init__(self):
        self.weights = {}

    def reset_weights(self):
        self.weights.clear()
        for p in PRIMITIVES:
            self.weights[p] = None

    def set_default_weights(self):
        self.reset_weights()
        self.set_weight('none', 1)
        self.set_weight('max_pool_3x3', 2)
        self.set_weight('avg_pool_3x3', 2)
        self.set_weight('skip_connect', 2)
        self.set_weight('sep_conv_3x3', 2)
        self.set_weight('sep_conv_5x5', 2)
        self.set_weight('dil_conv_3x3', 2)
        self.set_weight('dil_conv_5x5', 2)

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


class CustomLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=-100, reduce=True, current_alpha=None, oracle=None):
        super(CustomLoss, self).__init__(weight, size_average)
        self.current_network_cells_alphas = current_alpha
        self.oracle = oracle

    def update_current_network_cells_alphas(self, network_cells_alphas):
        self.current_network_cells_alphas = network_cells_alphas

    def forward(self, input, target):
        a = super(CustomLoss, self).forward(input, target)
        op_rate = self.oracle.get_operation_rate(self.current_network_cells_alphas) if self.oracle else 1.0
        return a * op_rate
