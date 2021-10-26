import thop
import torch

import genotypes
from model import NetworkCIFAR


def get_macs(network):
    inpt = torch.autograd.Variable(torch.randn(1, 3, 32, 32).cuda())
    macs, _ = thop.profile(network, inputs=(inpt,))
    return macs


def main():
    macs = {}
    assert type(genotypes.M1) == genotypes.Genotype
    for arch_id in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]:
        arch_id_str = 'M{}'.format(arch_id)
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id_str))).cuda()
        network.drop_path_prob = 0.0
        n_macs = get_macs(network)
        macs[arch_id_str] = n_macs
    for mac in macs:
        print('{}: {}'.format(mac, macs[mac]))


if __name__ == '__main__':
    main()
