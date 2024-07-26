import os
from datetime import datetime

import google.cloud.aiplatform as aip

task = "lfnd-99"
ts = datetime.now().strftime("%m%d-%H%M%S")
job_display_name = f"{task}-{ts}-allenoise-prod15-baseline"
job_dir = f"gs://kraken-anchor/jobs/{task}/{job_display_name}"

job_args = [
    "--batch-size=256",
    f"--job-dir={job_dir}",
    "--learning-rate=1e-4",
    "--num-epochs=10",
    "--model-path=gs://kraken-anchor/learning-from-noisy-data/encoders/xlm-roberta-base",
    "--tokenizer-path=gs://kraken-anchor/learning-from-noisy-data/encoders/xlm-roberta-base",
    "--train-file-path=gs://kraken-anchor/learning-from-noisy-data/real_world_dataset/v4/noisy/15/cv/0/train",
    "--val-file-path=gs://kraken-anchor/learning-from-noisy-data/real_world_dataset/v4/noisy/15/cv/0/val",
    "--test-file-path=gs://kraken-anchor/learning-from-noisy-data/real_world_dataset/v4/noisy/15/cv/0/test",
    "--seed=42",
    "--loss=cross-entropy",
    "--checkpoint-save-frequency-fraction=1",
    "--validation-sample-size=1",
    "--lfnd-logging-enabled",
]

gpu_count = 1
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": f"a2-ultragpu-{gpu_count}g",
            "accelerator_type": "NVIDIA_A100_80GB",
            "accelerator_count": gpu_count,
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310",
            "package_uris": ["gs://kraken-anchor/learning-from-noisy-data/packages/allenoise/category_classifier-0.0.1-py3-none-any.whl"],
            "python_module": "category_classifier.bert_classifier.train",
            "args": job_args,
        },
    }
]

aip.init()

job = aip.CustomJob(
    display_name=job_display_name,
    worker_pool_specs=worker_pool_specs,
    base_output_dir=job_dir,
    staging_bucket=os.path.join(job_dir, "staging"),
    location="europe-west4",
    labels={
        "team": "research",
        "project": "lfnd",
        "user": os.environ["USER"].replace(".", "-"),
        "env": "dev",
        "task-id": task,
    },
)

job.submit(service_account="698723338808-compute@developer.gserviceaccount.com")