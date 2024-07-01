import logging
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel

from abstract_classifier.encoder import AbstractEncoder
from bert_classifier.config.constants import TokenizerConstants
from bert_classifier.config.defaults import BertEncoderDefaults
from utils.io_utils import list_filepaths, save_json, translate_gcs_path_to_local


logger = logging.getLogger(__name__)


class BertEncoder(AbstractEncoder):
    """
    Class defining BERT encoder, used both for training and prediction.
    """

    def __init__(
        self,
        num_labels: int,
        encoder_dir: str,
        encoder_filename: str = BertEncoderDefaults.ENCODER_NAME,
        initialize_random_model_weights: bool = False,
    ) -> None:
        super().__init__()

        self._encoder_filename = encoder_filename
        self._encoder_dir = translate_gcs_path_to_local(encoder_dir)
        self._num_labels = num_labels

        bert_config = AutoConfig.from_pretrained(self._encoder_dir)
        if not initialize_random_model_weights and self._are_encoder_files_in_dir(self._encoder_dir):
            logger.info("Loading pretrained encoder from files.")
            self._bert = AutoModel.from_pretrained(self._encoder_dir, config=bert_config)
        else:
            logger.info("Creating randomly initialized encoder from config.")
            self._bert = AutoModel.from_config(config=bert_config)
        self._bert_output_embedding_size = bert_config.hidden_size
        self._fc = torch.nn.Linear(self._bert_output_embedding_size, self._num_labels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        mean_activation = self.mean_activation_of_the_last_hidden_state(input_tensor)
        return self._fc(mean_activation)

    def mean_activation_of_the_last_hidden_state(self, input_tensor: torch.Tensor) -> torch.Tensor:
        non_pad_tokens_mask = input_tensor != TokenizerConstants.PAD_TOKEN_ID
        model_output = self._bert(
            input_tensor, attention_mask=non_pad_tokens_mask
        )  # type: BaseModelOutputWithPoolingAndCrossAttentions

        masked_last_hidden_state = model_output.last_hidden_state * non_pad_tokens_mask.unsqueeze(-1)
        input_length = non_pad_tokens_mask.sum(dim=1, keepdim=True)
        mean_activation = masked_last_hidden_state.sum(dim=1) / input_length
        return mean_activation

    def dump_encoder_weights(self, parent_path: str, filename: str) -> None:
        super().dump_encoder_weights(parent_path, self._encoder_filename)
        self._bert.save_pretrained(parent_path)
        save_json(self._bert.config.to_dict(), parent_path, BertEncoderDefaults.CONFIG_NAME)

    def load_weights_from_checkpoint(
        self, parent_path: str, filename: str, device: str = BertEncoderDefaults.DEVICE
    ) -> None:
        super().load_weights_from_checkpoint(self._encoder_dir, self._encoder_filename, device)

    @staticmethod
    def _are_encoder_files_in_dir(encoder_dir: str) -> bool:
        encoder_files = [f for f in list_filepaths(encoder_dir, check_filename_extension=False)]
        encoder_filenames = [Path(f).name for f in encoder_files]
        are_encoder_files_in_dir = (
            BertEncoderDefaults.ENCODER_NAME in encoder_filenames
            or BertEncoderDefaults.PYTORCH_BIN_FILE_NAME in encoder_filenames
        )
        return are_encoder_files_in_dir
