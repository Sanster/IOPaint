import torch
from ..utils import torch_gc


class CPUTextEncoderWrapper(torch.nn.Module):
    def __init__(self, text_encoder, torch_dtype):
        super().__init__()
        self.config = text_encoder.config
        self.text_encoder = text_encoder.to(torch.device("cpu"), non_blocking=True)
        self.text_encoder = self.text_encoder.to(torch.float32, non_blocking=True)
        self.torch_dtype = torch_dtype
        del text_encoder
        torch_gc()

    def __call__(self, x, **kwargs):
        input_device = x.device
        return [
            self.text_encoder(x.to(self.text_encoder.device), **kwargs)[0]
            .to(input_device)
            .to(self.torch_dtype)
        ]

    @property
    def dtype(self):
        return self.torch_dtype
