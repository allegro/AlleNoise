from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


DATASET_DIRECTORY = f"./allenoise_cv"
FULL_DATASET_PATH = "../allenoise/full_dataset.csv"
VALID_FRACTION = 0.1


@dataclass
class FoldPaths:
    train: str
    val: str
    test: str

    @staticmethod
    def from_fold_id(fold_id: int, main_dir_name: str) -> FoldPaths:
        return FoldPaths(
            train=f"{main_dir_name}/{fold_id}/train/",
            val=f"{main_dir_name}/{fold_id}/val/",
            test=f"{main_dir_name}/{fold_id}/test/",
        )


def save_folds(df, dir_name, fold_count: int, is_clean: bool):
    if is_clean:
        selected_columns_list = ["offer_id", "text", "category_id"]
    else:
        selected_columns_list = ["offer_id", "text", "category_id", "category_id_true"]

    skf_indices = np.arange(0, df.shape[0], 1).reshape((df.shape[0], 1))
    skf_groups = df["category_id"].values.reshape((df.shape[0], 1))

    stratified_splitter = StratifiedKFold(n_splits=fold_count, shuffle=True)
    splits = stratified_splitter.split(skf_indices, skf_groups)

    for fold_id, (train_and_valid_indices, test_indices) in enumerate(splits):
        fold_train_plus_valid_df = df.iloc[train_and_valid_indices][selected_columns_list]

        fold_train_df, fold_valid_df = train_test_split(
            fold_train_plus_valid_df,
            test_size=VALID_FRACTION,
            train_size=1 - VALID_FRACTION,
            random_state=None,
            shuffle=True,
            stratify=fold_train_plus_valid_df["category_id"].values,
        )

        fold_test_df = df.iloc[test_indices][["offer_id", "text", "category_id_true"]].rename(
            columns={"category_id_true": "category_id"}
        )

        fold_train_df = fold_train_df[selected_columns_list]
        fold_valid_df = fold_valid_df[selected_columns_list]

        fold_paths = FoldPaths.from_fold_id(fold_id=fold_id, main_dir_name=dir_name)

        fold_train_df["offer_id"] = fold_train_df.apply(lambda row: str(int(row.offer_id)), axis=1)
        fold_valid_df["offer_id"] = fold_valid_df.apply(lambda row: str(int(row.offer_id)), axis=1)
        fold_test_df["offer_id"] = fold_test_df.apply(lambda row: str(int(row.offer_id)), axis=1)

        for outdir in [fold_paths.train, fold_paths.val, fold_paths.test]:
            if not os.path.exists(outdir):
                os.makedirs(outdir)

        fold_train_df.to_csv(os.path.join(fold_paths.train, "offers.csv"), sep="\t", index=False)
        fold_valid_df.to_csv(os.path.join(fold_paths.val, "offers.csv"), sep="\t", index=False)
        fold_test_df.to_csv(os.path.join(fold_paths.test, "offers.csv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-count", default=5, type=int, required=False)
    args = parser.parse_args()

    full_dataset = pd.read_csv(FULL_DATASET_PATH, sep="\t").rename(columns={"clean_category_id": "category_id_true"})

    if not os.path.exists(f"{DATASET_DIRECTORY}/clean/cv"):
        full_dataset["category_id"] = full_dataset["category_id_true"]
        save_folds(
            df=full_dataset,
            dir_name=f"{DATASET_DIRECTORY}/clean/cv",
            fold_count=args.fold_count,
            is_clean=True
        )

    if not os.path.exists(f"{DATASET_DIRECTORY}/noisy/15/cv"):
        full_dataset["category_id"] = full_dataset["noisy_category_id"]
        save_folds(
            df=full_dataset,
            dir_name=f"{DATASET_DIRECTORY}/noisy/15/cv",
            fold_count=args.fold_count, is_clean=False
        )