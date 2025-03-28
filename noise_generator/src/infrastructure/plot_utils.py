import os

import matplotlib.pyplot as plt
import pandas as pd

from infrastructure.io_utils import isdir, makedirs


def plot_category_size_distribution(category_counts: pd.DataFrame, output_dir: str, output_file: str) -> None:
    x = category_counts["count"].tolist()
    figure = plt.figure(figsize=(14, 4))
    plt.hist(x, bins=150)
    plt.yscale("log")
    plt.xlabel("category size")
    plt.ylabel("number of categories")
    plt.grid(True, which="both", axis="both")

    if not isdir(output_dir):
        makedirs(output_dir)
    target_file = os.path.join(output_dir, output_file)
    figure.savefig(target_file)
