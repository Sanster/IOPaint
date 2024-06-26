import unittest
import torch

from iopaint.model.anytext.ocr_recog.common import Activation

class TestActivation(unittest.TestCase):
    def test_activation_relu(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('relu')
        y = activation(x)
        self.assertTrue(torch.equal(y, torch.relu(x)))

    def test_activation_relu6(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('relu6')
        y = activation(x)
        self.assertTrue(torch.equal(y, torch.nn.functional.relu6(x)))

    def test_activation_sigmoid(self):
        with self.assertRaises(NotImplementedError):
            Activation('sigmoid')

    def test_activation_hardsigmoid(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('hard_sigmoid')
        y = activation(x)
        self.assertTrue(torch.equal(y, torch.nn.Hardsigmoid(x)))

    def test_activation_hardswish(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('hard_swish')
        y = activation(x)
        self.assertTrue(torch.equal(y, torch.nn.Hardswish(x)))

    def test_activation_leakyrelu(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('leakyrelu')
        y = activation(x)
        self.assertTrue(torch.equal(y, torch.nn.functional.leaky_relu(x)))

    def test_activation_gelu(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('gelu')
        y = activation(x)
        self.assertTrue(torch.equal(y, torch.nn.functional.gelu(x)))

    def test_activation_swish(self):
        x = torch.tensor([-1.0, 0.0, 1.0])
        activation = Activation('swish')
        y = activation(x)
        self.assertTrue(torch.equal(y, x * torch.sigmoid(x)))
    
    def test_activation_invalid_type(self):
        with self.assertRaises(NotImplementedError):
            Activation('none')

if __name__ == '__main__':
    unittest.main()