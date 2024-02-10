import torch
from transformers import PreTrainedModel

from ..utils import torch_gc


class CPUTextEncoderWrapper(PreTrainedModel):
    def __init__(self, text_encoder, torch_dtype):
        super().__init__(text_encoder.config)
        self.config = text_encoder.config
        self._device = text_encoder.device
        # cpu not support float16
        self.text_encoder = text_encoder.to(torch.device("cpu"), non_blocking=True)
        self.text_encoder = self.text_encoder.to(torch.float32, non_blocking=True)
        self.torch_dtype = torch_dtype
        del text_encoder
        torch_gc()

    def __call__(self, x, **kwargs):
        input_device = x.device
        original_output = self.text_encoder(x.to(self.text_encoder.device), **kwargs)
        for k, v in original_output.items():
            if isinstance(v, tuple):
                original_output[k] = [
                    v[i].to(input_device).to(self.torch_dtype) for i in range(len(v))
                ]
            else:
                original_output[k] = v.to(input_device).to(self.torch_dtype)
        return original_output

    @property
    def dtype(self):
        return self.torch_dtype

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return self._device