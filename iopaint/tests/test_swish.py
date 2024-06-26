import unittest
import torch

from iopaint.model.anytext.ocr_recog.common import Swish

class TestSwish(unittest.TestCase):
    def test_swish_inplace(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        swish = Swish(inplace=True)
        y = swish(x)
        self.assertTrue(torch.equal(y, x))

    def test_swish_not_inplace(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        swish = Swish(inplace=False)
        y = swish(x)
        self.assertTrue(torch.equal(y, x * torch.sigmoid(x)))

if __name__ == '__main__':
    unittest.main()
