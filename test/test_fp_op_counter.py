import unittest

from op_oracle import FPOpCounter


class TestFlopCounter(unittest.TestCase):
    def setUp(self):
        self.fc = FPOpCounter()

    def test_conv2d_2x2(self):
        flops = FPOpCounter.conv2d(3, 3, 3, 1, 2, 1, 0, 1, 1, False)
        self.assertEqual(48, flops)

    def test_conv2d_2x2_stride(self):
        flops = FPOpCounter.conv2d(5, 5, 3, 1, 2, 2, 0, 1, 1, False)
        self.assertEqual(48, flops)

    def test_conv2d_2x2_padding(self):
        flops = FPOpCounter.conv2d(3, 3, 3, 1, 2, 1, 1, 1, 1, False)
        self.assertEqual(192, flops)

    def test_conv2d_2x2_dilation(self):
        flops = FPOpCounter.conv2d(7, 7, 3, 1, 2, 1, 0, 2, 1, False)
        self.assertEqual(300, flops)

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
