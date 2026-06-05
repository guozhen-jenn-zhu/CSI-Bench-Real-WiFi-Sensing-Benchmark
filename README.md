# WiFi Sensing Benchmark

[Paper](https://arxiv.org/abs/2505.21866) | [Project page](https://ai-iot-sensing.github.io/projects/project.html) | [Paper with code](https://paperswithcode.com/paper/csi-bench-a-large-scale-in-the-wild-dataset)

A comprehensive benchmark and training system for WiFi sensing using CSI data. Accepted and presented at [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/121605).

## Overview

This repository provides a unified framework for training and evaluating deep learning models on WiFi Channel State Information (CSI) data for various sensing tasks. The framework supports both local execution and cloud-based training on AWS SageMaker.

## Installation and Setup

### Prerequisites

- Python 3.7+
- GPU Support (recommended, but not required):
  - NVIDIA GPU with CUDA support
  - Apple Silicon with MPS (Metal Performance Shaders)
  - CPU-only mode is available but much slower for training




### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/WiAL-Real-WiFi-Sensing-Benchmark.git
   cd WiAL-Real-WiFi-Sensing-Benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If you want to run multitask pipeline, please also install peft. As peft have version conflict in sagemaker instance, we didn't include that in requrirements.txt.  
   ```bash
   pip install peft
   ```



3. Data Download:

   Please find the [CSI-Bench dataset](https://www.kaggle.com/datasets/guozhenjennzhu/csi-bench) hosted on Kaggle, or pull it directly from our S3 bucket:

   ```bash
   # Recommended: mirror the whole dataset to ~/Data/CSI-Bench/ on your g5.12xlarge
   aws s3 sync s3://rnd-sagemaker/Data/fm_downstream/CSI-Bench/ ~/Data/CSI-Bench/
   ```

   The updated CSI-Bench layout no longer uses the legacy `tasks/` prefix.  Single-task datasets live directly under the root and the three multi-task sub-tasks share a `Multitask/` parent directory (so each per-task metadata can reference `../../sub_Human_h5/...`):

  ```
  CSI-Bench/
  ├── FallDetection/             # metadata/, splits/, sub_Human/
  ├── BreathingDetection/        # metadata/, splits/, sub_Human/
  ├── Localization/              # metadata/, splits/, sub_Human/
  ├── MotionSourceRecognition/   # metadata/, splits/, sub_{Human,Pet,IRobot,Fan}/
  └── Multitask/
      ├── HumanActivityRecognition/  # metadata/, splits/
      ├── HumanIdentification/       # metadata/, splits/
      ├── ProximityRecognition/      # metadata/, splits/
      ├── sub_Human_h5/              # shared H5 referenced via ../../ from metadata
      └── sub_Human_mat/
  ```

  Each task directory follows a consistent structure:
  ```
  TaskName/
  ├── sub_Human/                    # Contains all user data
  │   ├── user_U01/                 # Data for specific user
  │   │   ├── act_ActivityName/     # Data for specific activity
  │   │   │   ├── env_E01/          # Data from specific environment
  │   │   │   │   ├── device_DeviceName/  # Data from specific device
  │   │   │   │   │   └── session_TIMESTAMP__freqFREQ.h5  # Individual CSI recordings
  │   ├── user_U02/
  │   └── ...
  ├── metadata/                     # Metadata for the task
  │   ├── sample_metadata.csv       # Detailed information about each sample
  │   └── label_mapping.json        # Maps activity labels to indices
  └── splits/                       # Dataset splits for experiments
      ├── train_id.json             # Training set IDs
      ├── val_id.json               # Validation set IDs
      ├── test_id.json              # Test set IDs
      ├── test_easy[_id].json       # Easy difficulty test set
      ├── test_medium[_id].json     # Medium difficulty test set
      ├── test_hard[_id].json       # Hard difficulty test set
      ├── test_cross_device.json    # Out-of-distribution test (multitask only)
      ├── test_cross_env.json
      └── test_cross_user.json
  ```

  > Split files whose name contains `_p<digit>` (e.g. `train_id_p5.json`, `p5_info.json`) belong to a separate few-shot subset experiment and are automatically ignored by the loader / runner — you can keep them in place.


## Local Execution (supervised learning)

The main entry point for local execution is `scripts/local_runner.py`. This script handles configuration loading, model training, and result storage. For paper reproduction we ship `configs/csi_bench_local_config.json` which is preconfigured for the new dataset layout and a `g5.12xlarge` host.

### Configuration

Edit the local configuration file at `configs/csi_bench_local_config.json` to set your data path and other parameters:

```json
{
  "pipeline": "supervised",
  "training_dir": "~/Data/CSI-Bench/",
  "output_dir": "./results/csi_bench",
  "task": "FallDetection",
  "available_tasks": ["FallDetection", "BreathingDetection", "Localization", "MotionSourceRecognition"],
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"],
  "win_len": 500,
  "feature_size": 232,
  "seed": 42,
  "seeds": [42],
  "batch_size": 128,
  "epochs": 100,
  "learning_rate": 1e-3,
  "weight_decay": 1e-5,
  "warmup_epochs": 5,
  "patience": 15,
  "test_splits": "all"
}
```

Key parameters:
- `pipeline`: Training pipeline type
- `training_dir`: Root directory of your CSI-Bench dataset (e.g. `~/Data/CSI-Bench/`). The new layout no longer uses a `tasks/` prefix; the loader searches `<training_dir>/<task>/` and `<training_dir>/Multitask/<task>/` automatically.
- `output_dir`: Directory to save results (default: `./results/csi_bench`)
- `available_models`: Model types to train, default list is all 7 models in this project
- `available_tasks`: When set, the runner sweeps over each task in turn (otherwise it falls back to the single `task` field)
- `seed` / `seeds`: Pass a single seed (default 42 for the Phase E single-seed sweep) or a list (e.g. `[42, 43, 44]` for the Phase F mean ± std re-run)
- `batch_size`, `epochs`, `learning_rate`, `weight_decay`, `warmup_epochs`, `patience`: Training hyper-parameters (defaults match the paper appendix B.3)

### Running Models

Basic usage (single task, single seed):
```bash
python scripts/local_runner.py --config configs/csi_bench_local_config.json
```

Full paper-style sweep (all 4 single-tasks × 7 models × N seeds):
```bash
# Phase E -- single-seed sweep (seed=42).  Default deliverable.
python scripts/run_seed_sweep.py --config configs/csi_bench_local_config.json

# Phase F (opt-in) -- mean ± std with three seeds
python scripts/run_seed_sweep.py --config configs/csi_bench_local_config.json --seeds 42,43,44
```

### Available Models

- `mlp`: Multi-Layer Perceptron
- `lstm`: Long Short-Term Memory
- `resnet18`: ResNet-18 CNN
- `transformer`: Transformer-based model
- `vit`: Vision Transformer
- `patchtst`: PatchTST (Patch Time Series Transformer)
- `timesformer1d`: TimesFormer for 1D signals

### Available Tasks (Make sure you downloaded the whole dataset for corresponding task)

- `MotionSourceRecognition`
- `BreathingDetection_Subset`
- `Localization`
- `FallDetection`
- `ProximityRecognition`
- `HumanActivityRecognition`
- `HumanIdentification`




## Results Organization

Training results are saved with the following structure:

```
results/
├── task_name/                 # Name of the task
│   ├── model_name/            # Name of the model
│   │   ├── best_performance.json     # Record of best performance
│   │   ├── params_hash/              # Experiment identifier
│   │   │   ├── model_task_config.json           # Model configuration
│   │   │   ├── model_task_results.json          # Training metrics
│   │   │   ├── model_task_summary.json          # Performance summary
│   │   │   ├── model_task_test_confusion.png    # Confusion matrix
│   │   │   ├── classification_report_test.csv   # Classification metrics
│   │   │   └── checkpoint/                      # Saved model weights
│   │   └── 
│   |
│   └── 
└── ...
```



## Multi-Task Learning

The multi-task learning pipeline uses the same entry point as supervised learning: `scripts/local_runner.py`. This script handles configuration loading, training multiple tasks simultaneously, and organizing results.

### Configuration

The repo ships `configs/csi_bench_multitask_config.json` preconfigured for the paper's multi-task Transformer experiment (Tab. 4 / Tab. 12-14):

```json
{
  "pipeline": "multitask",
  "training_dir": "~/Data/CSI-Bench/",
  "output_dir": "./results/csi_bench_multitask",
  "model": "transformer",
  "tasks": ["HumanActivityRecognition", "HumanIdentification", "ProximityRecognition"],
  "feature_size": 232,
  "win_len": 500,
  "seed": 42,
  "seeds": [42],
  "batch_size": 128,
  "epochs": 100,
  "emb_dim": 128,
  "dropout": 0.1,
  "test_splits": "test_id,test_cross_device,test_cross_env,test_cross_user",
  "learning_rate": 5e-4,
  "weight_decay": 1e-5,
  "patience": 15,
  "lora_r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "available_models": ["transformer"]
}
```

Key parameters:
- `pipeline`: Set to `"multitask"` for the multi-task learning pipeline
- `training_dir`: Root of your CSI-Bench dataset.  The loader automatically searches `<training_dir>/Multitask/<task>/` for each multi-task sub-task.
- `output_dir`: Directory to save results (default: `./results/csi_bench_multitask`)
- `model`: Model type, currently multi-task learning supports `transformer`, `patchtst`, and `timesformer1d`
- `tasks`: List of tasks to train simultaneously
- `seed` / `seeds`: Single seed (default 42) or list for multi-seed runs
- `lora_r`, `lora_alpha`, `lora_dropout`: LoRA adapter parameters
- `learning_rate`: Learning rate, default is 5e-4
- `patience`: Early stopping patience value, default is 15

### Running Models

Basic usage:
```bash
python scripts/local_runner.py --config configs/csi_bench_multitask_config.json
```

…or via the seed sweep helper:

```bash
python scripts/run_seed_sweep.py --config configs/csi_bench_multitask_config.json
```

### Supported Models

Multi-task learning currently supports these models:
- `transformer`: Transformer-based model
- `patchtst`: PatchTST (Patch Time Series Transformer)
- `timesformer1d`: TimesFormer for 1D signals

### Available Tasks

Multi-task learning can train multiple tasks simultaneously. Make sure the specified tasks exist in your dataset:
- `MotionSourceRecognition`
- `BreathingDetection_Subset`
- `Localization`
- `FallDetection`
- `ProximityRecognition`
- `HumanActivityRecognition`
- `HumanIdentification`

### Benefits of Multi-Task Learning

Multi-task learning trains on multiple related tasks simultaneously by sharing underlying representations. This approach offers several advantages:
1. **Better Generalization**: By training on multiple tasks, the model learns more robust feature representations
2. **Improved Sample Efficiency**: Tasks with limited data can borrow knowledge from related tasks
3. **Faster Training**: Joint training is usually faster than training multiple separate models

Multi-task learning uses LoRA (Low-Rank Adaptation) technology to enable efficient multi-task learning with only a small number of task-specific parameters.



## SageMaker Integration

The repository provides robust support for scaling CSI-Bench training on AWS SageMaker.  The default ready-made config (`configs/csi_bench_sagemaker_config.json`) targets `ml.g5.12xlarge` instances and the new S3 layout at `s3://rnd-sagemaker/Data/fm_downstream/CSI-Bench/` — no `tasks/` prefix.

### Configuration

Edit `configs/csi_bench_sagemaker_config.json` to set your S3 paths and training parameters:

```json
{
  "pipeline": "supervised",
  "s3_data_base": "s3://rnd-sagemaker/Data/fm_downstream/CSI-Bench/",
  "s3_output_base": "s3://rnd-sagemaker/CSI-Bench-Results/",
  "win_len": 500,
  "feature_size": 232,
  "batch_size": 128,
  "epochs": 100,
  "learning_rate": 1e-3,
  "weight_decay": 1e-5,
  "instance_type": ["ml.g5.12xlarge"],
  "framework_version": "2.0.0",
  "py_version": "py310",
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"],
  "available_tasks": ["FallDetection", "BreathingDetection", "Localization", "MotionSourceRecognition"],
  "test_splits": "all",
  "use_root_data_path": true
}
```

Key parameters:
- `pipeline`: Training pipeline type (`supervised` or `multitask`)
- `s3_data_base`: S3 prefix that contains `FallDetection/`, `BreathingDetection/`, `Localization/`, `MotionSourceRecognition/`, and `Multitask/`.  The SageMaker runner mounts the right sub-prefix for each task automatically — single-tasks mount `<base>/<task>/`, multi-task sub-tasks mount `<base>/Multitask/` (so the shared `sub_Human_h5/` is available).
- `s3_output_base`: S3 path for storing results
- `instance_type`: AWS instance type (default `ml.g5.12xlarge`, 4×A10G + 192 GiB RAM)
- `available_models`: Model types to train
- `available_tasks`: One SageMaker job is submitted per task; each job sweeps all models

### Data Structure on S3

The expected S3 layout matches the new CSI-Bench layout described in the [Data Download](#installation-and-setup) section above:

```
s3://your-bucket/path/CSI-Bench/
├── FallDetection/
├── BreathingDetection/
├── Localization/
├── MotionSourceRecognition/
└── Multitask/
    ├── HumanActivityRecognition/
    ├── HumanIdentification/
    ├── ProximityRecognition/
    ├── sub_Human_h5/
    └── sub_Human_mat/
```

### Running Models

Basic usage:
```bash
python scripts/sagemaker_runner.py --config configs/csi_bench_sagemaker_config.json
```




### Batch Processing

The SageMaker runner supports batch processing to run multiple tasks and models. It will automatically create separate training jobs for each task, using all models specified in the configuration.



### Job Management

Training jobs are submitted to SageMaker in non-blocking mode. You can monitor their progress in the AWS SageMaker console or use the AWS CLI.

Results and model artifacts will be stored in the S3 output location you specified in the configuration file.

### Advantages of SageMaker Integration

1. **Scalability**: Train on powerful GPU instances without local hardware constraints
2. **Parallelization**: Run multiple experiments simultaneously
3. **Cost Efficiency**: Only pay for the compute time you use
4. **Reproducibility**: Consistent environment for all experiments




## Reproducing the CSI-Bench paper

The repo ships three preconfigured configs and a sweep helper that reproduce every table in the NeurIPS 2025 paper (Tab. 3, 4, 8-14) on a `ml.g5.12xlarge` instance against the updated dataset at `s3://rnd-sagemaker/Data/fm_downstream/CSI-Bench/` (or its local mirror `~/Data/CSI-Bench/`).

### One-time setup

```bash
# 1) Mirror the dataset locally on the g5.12xlarge (~78 GiB)
aws s3 sync s3://rnd-sagemaker/Data/fm_downstream/CSI-Bench/ ~/Data/CSI-Bench/

# 2) Install dependencies and (only for multi-task LoRA) install peft
pip install -r requirements.txt
pip install peft

# 3) (one-time) regenerate the 4-class MotionSourceRecognition mapping if needed
python util/build_msr_label_mapping.py --verify
```

### Phase E — single-seed sweep (default deliverable)

```bash
# Supervised single-task (Tab. 3 + difficulty tables Tab. 8/9/10/11):
python scripts/run_seed_sweep.py --config configs/csi_bench_local_config.json

# Multi-task adapter (Tab. 4 multi-task vs single-task) and OOD tables (Tab. 12/13/14):
python scripts/run_seed_sweep.py --config configs/csi_bench_multitask_config.json
```

### Phase F — three-seed sweep (optional, opt-in)

```bash
python scripts/run_seed_sweep.py --config configs/csi_bench_local_config.json     --seeds 42,43,44
python scripts/run_seed_sweep.py --config configs/csi_bench_multitask_config.json --seeds 42,43,44
```

### Aggregating results into paper tables

`result_analysis/all_result_summary.py` walks the results directory and emits the paper-style CSV tables.  Pass `--seeds 42` (default) for single-seed reporting or `--seeds 42,43,44` for the mean ± std format used in the paper.

```bash
# Single-seed (Phase E)
python result_analysis/all_result_summary.py \
  --results-dir ./results/csi_bench \
  --multitask-results-dir ./results/csi_bench_multitask \
  --output-dir result_analysis/csi_bench_new

# Three seeds (Phase F)
python result_analysis/all_result_summary.py \
  --results-dir ./results/csi_bench \
  --multitask-results-dir ./results/csi_bench_multitask \
  --output-dir result_analysis/csi_bench_new_3seeds \
  --seeds 42,43,44
```

Output CSVs (one per paper table):

| File | Paper reference |
|---|---|
| `tab3_supervised.csv` | Tab. 3 — single-task supervised accuracy / F1 on `test_id` |
| `tab4_multitask_vs_singletask.csv` | Tab. 4 — multitask Transformer vs single-task Transformer |
| `tab8_fall_difficulty.csv` | Tab. 8 — FallDetection easy / medium / hard |
| `tab9_breath_difficulty.csv` | Tab. 9 — BreathingDetection easy / medium / hard |
| `tab10_loc_difficulty.csv` | Tab. 10 — Localization easy / medium / hard |
| `tab11_msr_difficulty.csv` | Tab. 11 — MotionSourceRecognition easy / medium / hard (4-class) |
| `tab12_har_ood.csv` | Tab. 12 — HumanActivityRecognition cross-device / env / user |
| `tab13_uid_ood.csv` | Tab. 13 — HumanIdentification cross-device / env / user |
| `tab14_prox_ood.csv` | Tab. 14 — ProximityRecognition cross-device / env / user |




## Citation

If you use this code in your research, please cite:
```
@article{zhu2026csi,
  title={CSI-Bench: A large-scale in-the-wild dataset for multi-task WiFi sensing},
  author={Zhu, Guozhen and Hu, Yuqian and Gao, Weihang and Wang, Wei-Hsiang and Wang, Beibei and Liu, K},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  year={2026}
}
```

## License

This project is licensed under Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).
