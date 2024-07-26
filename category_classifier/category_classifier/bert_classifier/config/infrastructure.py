import typing as T

import torch

from category_classifier.utils.io_utils import is_cuda_available


class AmpConfig:
    FP_16_NATIVE_OPT = {"amp_backend": "native", "amp_level": None, "precision": 16}
    FP_32_NATIVE_OPT = {"amp_backend": "native", "amp_level": None, "precision": 32}

    FP_OPT_DEFAULT = FP_16_NATIVE_OPT if is_cuda_available() else FP_32_NATIVE_OPT


class InfrastructureConfig:
    # choose this for standard GCP ai-platform training (with GPU support)
    SINGLE_MACHINE_TRAINING_GPU = {"strategy": None, "replace_sampler_ddp": False, "num_nodes": 1, "gpus": 1}
    # choose this for local development on your machine
    SINGLE_MACHINE_TRAINING_CPU = {"strategy": None, "replace_sampler_ddp": False, "num_nodes": 1, "gpus": 0}

    # choose this for local development on your machine if you need to `simulate` ddp training
    DDP_LOCAL_DEVELOPMENT = {
        "strategy": "ddp",
        "accelerator": "cpu",
        "replace_sampler_ddp": False,
        "num_nodes": 1,
        "gpus": None,
    }
    # choose this for single machine training x 2 GPUs ddp training
    DDP_CONFIG_SINGLE_NODE = {"strategy": "ddp", "replace_sampler_ddp": False, "num_nodes": 1, "gpus": 2}
    # choose this for 4 machines with x 2 GPUs  ddp training
    gpu_count = torch.cuda.device_count()
    DDP_CONFIG_64_PARTITIONS = {"strategy": "ddp", "replace_sampler_ddp": False, "num_nodes": 1, "gpus": gpu_count}
    # add here other configurations

    DDP_CONFIG_DEFAULT = DDP_CONFIG_64_PARTITIONS if is_cuda_available() else DDP_LOCAL_DEVELOPMENT
    SINGLE_MACHINE_CONFIG_DEFAULT = SINGLE_MACHINE_TRAINING_GPU if is_cuda_available() else SINGLE_MACHINE_TRAINING_CPU


def compute_world_size(num_gpus: T.Optional[int], num_nodes: int):
    if num_gpus:
        world_size = num_nodes * num_gpus
    else:
        world_size = 1
    return world_size if is_cuda_available() else 1
