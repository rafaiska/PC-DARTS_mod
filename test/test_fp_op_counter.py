import unittest

import genotypes
from op_oracle import FLOPCounter


class TestFlopCounter(unittest.TestCase):
    def setUp(self):
        self.fc = FLOPCounter()

    def test_conv2d_2x2(self):
        flops = FLOPCounter.conv2d(3, 3, 3, 3, 2, 1, 0, 1, 1, False)
        self.assertEqual(52, flops)

    def test_conv2d_2x2_stride(self):
        flops = FLOPCounter.conv2d(5, 5, 3, 3, 2, 2, 0, 1, 1, False)
        self.assertEqual(0, flops)

    def test_conv2d_2x2_padding(self):
        flops = FLOPCounter.conv2d(3, 3, 3, 3, 2, 1, 2, 1, 1, False)
        self.assertEqual(0, flops)

    def test_setup(self):
        self.fc.setup(8, 16)
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
        flops = self.fc.count_network_flops()
        print(flops)