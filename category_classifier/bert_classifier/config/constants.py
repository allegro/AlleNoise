class GlobalConstants:
    CATEGORY_MAPPING_DIR = "category_mapping"
    CATEGORY_MAPPING_FILENAME = "category_mapping.json"

    CSV_DELIMITER = "\t"
    CSV_ESCAPE_CHAR = "\\"


class BertTrainingConstants:
    LATEST_CHECKPOINT_FILENAME = "latest.ckpt"
    FINAL_CHECKPOINT_FILENAME = "final.ckpt"
    FINAL_PRODUCTION_CHECKPOINT_NAME = "encoder"
    CHECKPOINT_DIRNAME = "checkpoints"
    SELECTED_LABELS_KEY = "selected_labels"
    SELECTED_TRUE_LABELS_KEY = "selected_true_labels"

    def production_checkpoint_name(epoch_no: float) -> str:
        return f"encoder_after_{epoch_no:.2f}_epochs"
    

class AnchorBertDatasetConstants:
    OFFER_ID_COL_NAME = "offer_id"
    CATEGORY_ID_COL_NAME = "category_id"
    TRUE_CATEGORY_ID_COL_NAME = "category_id_true"
    TEXT_COL_NAME = "text"
    INDEX_COL_NAME = "index"

    OFFER_ID_FILED_NAME = "offer_id"
    LABEL_FIELD_NAME = "label"
    TRUE_LABEL_FIELD_NAME = "true_label"
    FEATURES_FIELD_NAME = "token_ids"
    INDEX_FIELD_NAME = "index"


class AnchorBertPredictorConstants:
    PREDICTIONS_OUTPUT_FILE_NAME = "predictions.csv"
    PREDICTIONS_FILE_HEADER = "offer_id,predicted_category,predicted_score\n"


class AnchorBertTokenizerConstants:
    PADDING = "max_length"
    MAX_LENGTH = 64


class TokenizerConstants:
    TOKENIZER_FILENAME = "tokenizer.json"
    PAD_TOKEN = "<pad>"
    PAD_TOKEN_ID = 1
    # TokenizerConstants.MAX_LENGTH must be the same as AnchorBertTokenizerConstants.MAX_LENGTH
    MAX_LENGTH = 64


class DataPartitioningConstants:
    TOKENIZED_BLOCK_PREFIX = "file_block"
    TOKENIZED_BLOCK_EXTENSION = ".json"
