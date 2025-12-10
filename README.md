# WiFi Sensing Benchmark

[Paper](https://arxiv.org/abs/2505.21866) | [Project page](https://ai-iot-sensing.github.io/projects/project.html) | [Paper with code](https://paperswithcode.com/paper/csi-bench-a-large-scale-in-the-wild-dataset)

A comprehensive benchmark and training system for WiFi sensing using CSI data.

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

   Our full dataset will be released to public by camera ready deadline of NeurIPS 2025. For reviewers, please use the link in the paper manuscript to access our data through Kaggle. After downloaded the dataset, it should be in the following :
    
  ```
  CSI-Bench/
  ├── HumanActivityRecognition/
  ├── FallDetection/
  ├── BreathingDetection/
  ├── Localization/
  ├── HumanIdentification/
  ├── MotionSourceRecognition/
  └── ProximityRecognition/
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
      ├── test_easy.json            # Easy difficulty test set
      ├── test_medium.json          # Medium difficulty test set
      └── test_hard.json            # Hard difficulty test set
  ```


## Local Execution (supervised learning)

The main entry point for local execution is `scripts/local_runner.py`. This script handles configuration loading, model training, and result storage.

### Configuration

Edit the local configuration file at `configs/local_default_config.json` to set your data path and other parameters:

```json
{
  "pipeline": "supervised",
  "training_dir": "/path/to/your/data/",
  "output_dir": "./results", 
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"],
  "task": "YourTask",
  "win_len": 500,
  "feature_size": 232,
  "batch_size": 32,
  "epochs": 100,
  "test_splits": "all"
}
```

Key parameters:
- `pipeline`: Training pipeline type 
- `training_dir`: Path to your data directory. The scripts will look for data at `training_dir/tasks/CURRENT_TASK/...`. Make sure that the data directory is the root directory where you downloaded the dataset. It should contain a "tasks" folder with multiple subfolders for different tasks. Examples:
  ```
  "C:\\Users\\weiha\\Desktop\\CSI-Bench"
  "/Users/leo/Desktop/CSI-Bench"
  ```
- `output_dir`: Directory to save results (default: `./results`)
- `available_models`: Model types to train, default list is all models in this project
- `task`: Task name (see Available Tasks)
- `batch_size`, `epochs`: Training parameters

### Running Models

Basic usage:
```bash
python scripts/local_runner.py
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

Modify configuration file for multi-task learning `configs/multitask_config.json`:

```json
{
  "pipeline": "multitask",
  "training_dir": "/path/to/your/data/",
  "output_dir": "./results",
  "model": "transformer",
  "tasks": ["TaskA", "TaskB"],
  "feature_size": 232,
  "win_len": 500,
  "batch_size": 32,
  "epochs": 30,
  "emb_dim": 128,
  "dropout": 0.1,
  "test_splits": "all",
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
- `training_dir`: Path to your data directory. The scripts will look for data at `training_dir/tasks/CURRENT_TASK/...`. Make sure that the data directory is the root directory where you downloaded the dataset. It should contain a "tasks" folder with multiple subfolders for different tasks. Examples:
  ```
  "C:\\Users\\weiha\\Desktop\\CSI-Bench"
  "/Users/leo/Desktop/CSI-Bench"
  ```
- `output_dir`: Directory to save results (default: `./results`)
- `model`: Model type, currently multi-task learning supports `transformer`, `patchtst`, and `timesformer1d`
- `tasks`: List of tasks to train simultaneously
- `lora_r`, `lora_alpha`, `lora_dropout`: Parameters for LoRA adapters
- `learning_rate`: Learning rate, default is 5e-4
- `patience`: Early stopping patience value, default is 15

### Running Models

Basic usage:
```bash
python scripts/local_runner.py --config_file configs/multitask_config.json
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

The repository provides robust support for scaling WiFi sensing model training on AWS SageMaker. This allows you to leverage cloud computing resources for larger experiments.

### Configuration

Edit the SageMaker configuration file at `configs/sagemaker_default_config.json` to set your S3 paths and training parameters:

```json
{
  "pipeline": "supervised",
  "s3_data_base": "s3://your-bucket/path/to/data/",
  "s3_output_base": "s3://your-bucket/path/to/output/",
  "win_len": 500,
  "feature_size": 232,
  "batch_size": 128,
  "epochs": 100,
  "learning_rate": 1e-3,
  "weight_decay": 1e-5,
  "instance_type": ["ml.g4dn.2xlarge"],
  "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit", "patchtst", "timesformer1d"],
  "available_tasks": ["YourTask1", "YourTask2"],
  "test_splits": "all"
}
```

Key parameters:
- `pipeline`: Training pipeline type (currently sagemaker runner only support `supervised` `)
- `s3_data_base`: S3 path to your data directory (must follow the expected structure)
- `s3_output_base`: S3 path for storing results
- `instance_type`: AWS instance type to use (can be a list for different tasks)
- `available_models`: Model types to train
- `available_tasks`: Tasks to run experiments on. It will submit 1 training job for one task (with all models in the list)

### Data Structure

You can store the benchmark dataset to S3
```
s3://your-bucket/path/to/data/
├── tasks/
│   ├── TaskName1/
│   │   ├── <standard task structure>
│   ├── TaskName2/
│   └── ...
```

### Running Models

Basic usage:
```bash
python scripts/sagemaker_runner.py
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




## Citation

If you use this code in your research, please cite:
```
@article{zhu2025csi,
  title={CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing},
  author={Zhu, Guozhen and Hu, Yuqian and Gao, Weihang and Wang, Wei-Hsiang and Wang, Beibei and Liu, KJ},
  journal={arXiv preprint arXiv:2505.21866},
  year={2025}
}
```

## License

This project is licensed under Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
