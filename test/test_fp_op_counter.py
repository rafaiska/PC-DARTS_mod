import unittest

import genotypes
from op_oracle import FPOpCounter


class TestFlopCounter(unittest.TestCase):
    def setUp(self):
        self.fc = FPOpCounter()

    def test_conv2d_2x2(self):
        flops = FPOpCounter.conv2d(3, 3, 3, 3, 2, 1, 0, 1, 1, False)
        self.assertEqual(52, flops)

    def test_conv2d_2x2_stride(self):
        flops = FPOpCounter.conv2d(5, 5, 3, 3, 2, 2, 0, 1, 1, False)
        self.assertEqual(0, flops)

    def test_conv2d_2x2_padding(self):
        flops = FPOpCounter.conv2d(3, 3, 3, 3, 2, 1, 2, 1, 1, False)
        self.assertEqual(0, flops)

    def test_setup(self):
        self.fc.setup(32, 32, 8, 16)
        self.assertEqual([(16, 16, (32, 32), (32, 32)),
                          (16, 16, (32, 32), (32, 32)),
                          (16, 32, (32, 32), (16, 16)),
                          (32, 32, (16, 16), (16, 16)),
                          (32, 32, (16, 16), (16, 16)),
                          (32, 64, (16, 16), (8, 8)),
                          (64, 64, (8, 8), (8, 8)),
                          (64, 64, (8, 8), (8, 8))],
                         self.fc.layers)

    def test_count(self):
        self.fc.setup(32, 32, 8, 16)
        self.fc.genotype = genotypes.M1
        flops = self.fc.count_network_fp_ops()
        print(flops)

    def test_count_genotypes(self):
        self.fc.setup(32, 32, 20, 36)
        for geno_id in [1, 2, 3, 5, 6, 7, 8, 9, 10]:
            self.fc.genotype = eval('genotypes.M{}'.format(geno_id))
            print('M{} fp ops: {}'.format(geno_id, self.fc.count_network_fp_ops()))
