import torch

import genotypes
import thop
from model import NetworkCIFAR
from scripts.arch_data import ArchDataCollection


def get_all_macs(rewrite=True):
    assert type(genotypes.M1) == genotypes.Genotype
    collection = ArchDataCollection(collection_file_path='/workspace/mount/.arch_data')
    collection.load()
    for arch_id in collection.archs:
        if not rewrite and collection.archs[arch_id].macs_count:
            continue
        print('Profiling arch {}'.format(arch_id))
        network = NetworkCIFAR(36, 10, 20, False, eval('genotypes.{}'.format(arch_id))).cuda()
        collection.archs[arch_id].macs_count = thop.profile(
            network, inputs=(torch.randn(1, 3, 32, 32).cuda(),), verbose=False)
        collection.save()
    return collection


if __name__ == '__main__':
    get_all_macs(True).csv_dump('/workspace/mount/arch_data.csv')
