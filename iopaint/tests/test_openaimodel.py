# added by me to run unittest on openaimodel.py
import unittest
import torch
import torch.nn as nn
from iopaint.model.anytext.ldm.modules.diffusionmodules.openaimodel import (
    TimestepEmbedSequential,
    TimestepBlock,
    SpatialTransformer,
    Upsample,
)

# simplified class for TimestepBlock
class MockTimestepBlock(TimestepBlock):
    def forward(self, x, emb):
        return x + emb

# simplifies attributes in SpatialTransformer
class MockSpatialTransformer(SpatialTransformer):
    def __init__(self, *args, **kwargs):
        # initialize nn.Module to bypass SpatialTransformer's __init__
        nn.Module.__init__(self)
        # placeholder attributes to mimic interface, but do nothing (nn.Identity output doesn't change input)
        self.proj_in = nn.Identity()
        self.norm = nn.Identity()
        self.transformer_blocks = nn.ModuleList([nn.Identity()])
        self.proj_out = nn.Identity()
        self.use_linear = True

    def forward(self, x, context=None):
        return x # non real transformation

# test cases for TimestepEmbedSequential
class TestTimestepEmbedSequential(unittest.TestCase):
    #  test behavior of an instance using a MockTimestepBlock
    def test_timestep_block_execution(self):
        seq = TimestepEmbedSequential(MockTimestepBlock())
        x = torch.tensor([[1., 2., 3.]])
        emb = torch.tensor([0.1, 0.2, 0.3])
        # call forward() of seq with x and emb
        output = seq(x, emb)
        expected_output = x + emb
        # check that TimestepEmbedSequential correctly applies the MockTimestepBlock to the inputs
        self.assertTrue(torch.equal(output, expected_output))

    #  test behavior of an instance using a MockSpatialTransformer
    def test_spatial_transformer_execution(self):
        seq = TimestepEmbedSequential(MockSpatialTransformer())
        # create randomised 4D tensors to simulate image-like data
        x = torch.randn(1, 3, 4, 4)
        context = torch.randn(1, 3, 4, 4)
        # call the forward() with context
        output = seq(x, None, context)
        expected_output = x
        self.assertTrue(torch.equal(output, expected_output))

    # tests behaviour of instance using a non-specialised layer (nn.Linear)
    def test_non_specialized_layer_execution(self):
        seq = TimestepEmbedSequential(nn.Linear(3, 3))
        x = torch.tensor([[1., 2., 3.]])
        emb = torch.tensor([0.1, 0.2, 0.3])
        # call forward() of seq
        output = seq(x, emb)
        self.assertFalse(torch.equal(output, x), "Output should not be identical to input")

class TestUpsample(unittest.TestCase):
    # test case for 3D without convolution
    def test_upsample_3d(self):
        upsample = Upsample(channels=3, use_conv=False, dims=3)
        x = torch.randn(1, 3, 4, 4, 4) # create random tensor shape (batch size, channels, depth, height, width)
        out = upsample(x) # upscale spatial dimensions
        expected_shape = (1, 3, 4, 8, 8) # expect height and width to double
        self.assertEqual(out.shape, expected_shape)

    # test case for 2D with convolution
    def test_upsample_2d(self):
        upsample = Upsample(channels=3, use_conv=True, dims=2)
        x = torch.randn(1, 3, 4, 4)
        out = upsample(x)
        expected_shape = (1, 3, 8, 8)
        self.assertEqual(out.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()