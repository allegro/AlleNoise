# AlleNoise

## What is AlleNoise?

**AlleNoise** is a large-scale text classification benchmark dataset with real-world label noise. It contains over 500,000 product titles taken from Allegro offers, with over 5,600 associated product categories. We provide two categories for each data point: 1) noisy category, initially set by sellers for their listed offer, 2) clean category, manually selected by human annotators as the correct one. In the noisy categories, there is effectively 15% of label noise present. With **AlleNoise**, we want to provide a benchmark dataset with realistic instance-dependent noise that could be used for the development of new robust classification methods.

## What is in this repo?

### AlleNoise

The **AlleNoise** dataset is stored in the `allenoise` directory. It contains the following files:
* full_dataset.csv
* category_mapping.csv
* metadata.json
* data_sheet.pdf

### Category classifier

Our category classification code that was used for all experiments. It includes re-implementations of several established methods for mitigating label noise. For details, consult the paper.

How to use:
* install the dependencies with Poetry:
```
cd category_classifier; poetry install
```
* split **AlleNoise** in a cross-validation scheme:
```
python prepare_dataset.py --fold-count FOLDS
```
where `FOLDS` is the number of cross-validation folds. 

The result of this script is a folder named `allenoise_cv`, which contains 2 directories: `clean` and `noisy`. The former contains cross-validation splits with clean training data, while the latter stores splits with real-world noise applied to training and validation data.
* start baseline training with default hyperparameter values:
```
python bert_classifier/train.py \
      --job-dir JOB_DIR \
      --model-path MODEL_PATH \
      --tokenizer-path TOKENIZER_PATH \
      --checkpoint-save-frequency-fraction 1 \
      --train-file-path TRAIN_FILE_PATH \
      --val-file-path VAL_FILE_PATH \
      --test-file-path TEST_FILE_PATH \
      --seed SEED \
      --num-epochs 10 \
      --validation-sample-size 1 \
      --batch-size 256 \
      --learning-rate 0.0001 \
      --loss cross-entropy \
      --lfnd-logging-enabled
```
where `JOB_DIR` is a path to a folder where training artifacts will be stored, `MODEL_PATH` is a path to a folder with XLMRoBERTa model weights, `TOKENIZER_PATH` is a path to the XLMRoBERTa tokenizer, `TRAIN_FILE_PATH` is a path to the training split, `VAL_FILE_PATH` is a path to the test split, `TEST_FILE_PATH` is the path to the test split and SEED is the random SEED to be set troughout the training.

To enable robust training, change the above command as follows (consult the paper for detailed method descriptions):
* Provably Robust Learning (PRL)
```
--loss prl-l-cross-entropy \
--prl-spl-coteaching-noise-level NOISE_LEVEL
```
where `NOISE_LEVEL` is a float in range (0, 1.0) indicating the expected noise level in the training data.
* Self-Paced Learning (SPL)
```
--loss spl-cross-entropy \
--prl-spl-coteaching-noise-level NOISE_LEVEL
```
where `NOISE_LEVEL` is a float in range (0, 1.0) indicating the expected noise level in the training data.
* Early Learning Regularization (ELR)
```
--loss elr-cross-entropy \
--elr-targets-momentum-beta BETA \
--elr-regularization-constant-lambda LAMBDA \
--elr-clamp-margin MARGIN
```
where `BETA`, `LAMBDA` and `MARGIN` are ELR hyperparameters.
* Generalized Jensen-Shannon Divergence (GJSD)
```
--loss gjsd-loss \
--gjsd-num-distributions M \
--gjsd-pi-weight PI
```
where `M` and `PI` are GJSD hyperparameters.
* Clipped Cross-Entropy (CCE)
```
--loss clipped-cross-entropy \
--cce-clip-loss-at-value CLIP \
--cce-start-from-epoch K
```
where `CLIP` and `K` are CCE hyperparameters.
* Mixup (MU)
```
--mixup \
--mixup-alpha ALPHA \
--mixup-ratio RATIO
```
where `ALPHA` and `RATIO` are MU hyperparameters.
* Co-Teaching (CT)
```
--co-teaching \
--co-teaching-variant CO_TEACHING \
--co-teaching-epoch-k K \
--prl-spl-coteaching-noise-level NOISE_LEVEL
```
where `NOISE_LEVEL` is a float in range (0, 1.0) indicating the expected noise level in the training data and `K` is a CT hyperparameter.
* Co-Teaching+ (CT+)
```
--co-teaching \
--co-teaching-variant CO_TEACHING_PLUS \
--co-teaching-epoch-k K \
--prl-spl-coteaching-noise-level NOISE_LEVEL
```
where `NOISE_LEVEL` is a float in range (0, 1.0) indicating the expected noise level in the training data and `K` is a CT+ hyperparameter.

### Noise generator

Our synthetic noise generation code. We used it to introduce synthetic noise into clean labels from **AlleNoise** for the purpose of comparing simple noise types to real-world noise provided with **AlleNoise**. For details regarding the noise types, consult the paper.

How to use:
* install the dependencies with Poetry:
```
cd noise_generator; poetry install
```
* generate symmetric noise:
```
python api/noise_generator_main.py --dataset-path DATASET_PATH --output-path OUTPUT_PATH --noise-type SYMMETRIC --noise-proportion NOISE_PROPORTION
```
* generate asymmetric pairflip noise:
```
python api/noise_generator_main.py --dataset-path DATASET_PATH --output-path OUTPUT_PATH --noise-type ASYMMETRIC_PAIRFLIP --noise-proportion NOISE_PROPORTION
```
* generate asymmetric nestedflip noise:
```
python api/noise_generator_main.py --dataset-path DATASET_PATH --output-path OUTPUT_PATH --noise-type ASYMMETRIC_NESTEDFLIP --noise-proportion NOISE_PROPORTION --category-path-mapping-file CATEGORY_PATH_MAPPING_FILE
```
where `CATEGORY_PATH_MAPPING_FILE` is the path to a file containing the mapping between category leaf IDs and category paths.
* generate asymmetric matrixflip noise:
```
python api/noise_generator_main.py --dataset-path DATASET_PATH --output-path OUTPUT_PATH --noise-type NOISE_TYPE --noise-proportion NOISE_PROPORTION --category-id-mapping-file CATEGORY_ID_MAPPING_FILE --model-predictions-file MODEL_PREDICTIONS_FILE
```
where `CATEGORY_ID_MAPPING_FILE` is the path to a file containing the mapping between category leaf IDs and token IDs, and `MODEL_PREDICTIONS_FILE` is the path to a file containing model predictions on the clean test set.

In all of the above, `DATASET_PATH` is the path to **AlleNoise**, `OUTPUT_PATH` is the path to the output file with perturbed categories and `NOISE_PROPORTION` is a float in range (0, 1.0) that indicates the desired noise level (e.g. for noise level 15%, this should be set to 0.15).

## How to cite us?

If you use AlleNoise, please cite the following paper:

```
Rączkowska, A., Osowska-Kurczab, A., Szczerbiński, J., Jasinska-Kobus, K., Nazarko, K., AlleNoise - large-scale text classification benchmark dataset with real-world label noise, 2024
```