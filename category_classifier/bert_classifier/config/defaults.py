from dataclasses import dataclass

from pcs_category_classifier.bert_classifier.domain_model.co_teaching import (
    CoTeachingPlusUpdateStrategy,
    CoTeachingVariant,
)
from pcs_category_classifier.bert_classifier.model.loss import Losses


class BertTrainingDefaults:
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.0
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    CHECKPOINT_SAVE_FREQUENCY_FRACTION = 0.25
    NUM_EPOCHS = 10
    VALID_SAMPLE_SIZE = 0.1
    TRAIN_STAGE_NAME = "train"
    LOSS = Losses.CROSS_ENTROPY
    RELOAD_DATALOADERS = 0
    FINE_THRESHOLD = 0.5
    FINE_BATCH_SIZE = 1024
    MIXUP_ALPHA = 0.2
    MIXUP_RATIO = 0.2
    GJSD_NUM_DISTRIBUTIONS = 4
    GJSD_PI_WEIGHT = 0.5
    GJSD_AUGMENTATIONS = "DROP_RANDOM-0.2,DROP_CONSECUTIVE-0.2,SWAP-0.3"
    CCE_CLIP_VALUE = 4.0
    CCE_START_EPOCH = 1.0
    CO_TEACHING_ENABLED = False
    CO_TEACHING_EPOCH = 1
    CO_TEACHING_VARIANT = CoTeachingVariant.CO_TEACHING
    CO_TEACHING_PLUS_UPDATE_STRATEGY = CoTeachingPlusUpdateStrategy.RECOMMENDED


class BertEvaluationDefaults:
    MC_DROPOUT_TRIALS = 10
    TEST_STAGE_NAME = "test"
    PREDICTIONS_FILE_NAME = "predictions.json"


class BertEncoderDefaults:
    ENCODER_NAME = "encoder.pt"
    PYTORCH_BIN_FILE_NAME = "pytorch_model.bin"
    CONFIG_NAME = "config.json"
    DEVICE = "cuda"


@dataclass
class TrainerLoggingConfig:
    progress_bar_refresh_rate: int = 0
    flush_logs_every_n_steps: int = 1000
    log_every_n_steps: int = 1000
    num_sanity_val_steps: int = 0
