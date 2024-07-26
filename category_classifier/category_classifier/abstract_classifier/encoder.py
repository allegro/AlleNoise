import os
from abc import ABC, abstractmethod

import torch

import category_classifier.utils.io_utils as io_utils
from category_classifier.bert_classifier.config.defaults import BertEncoderDefaults


class AbstractEncoder(torch.nn.Module, ABC):
    """
    Abstract class defining the encoder, used either for training and prediction.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ...

    def dump_encoder_weights(self, parent_path: str, filename: str) -> None:
        if not io_utils.isdir(parent_path):
            io_utils.makedirs(parent_path)
        io_utils.torch_save_state_dict(self.state_dict(), os.path.join(parent_path, filename))

    def load_weights_from_checkpoint(
        self, parent_path: str, filename: str, device: str = BertEncoderDefaults.DEVICE
    ) -> None:
        self.load_state_dict(
            io_utils.torch_load_state_dict(os.path.join(parent_path, filename), map_location=torch.device(device))
        )
