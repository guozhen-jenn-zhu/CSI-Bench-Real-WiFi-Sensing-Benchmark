#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Training Script - Train multiple model architectures in one training job

This script can be run in a SageMaker environment to train and evaluate multiple model architectures
on the same task.

SageMaker Training Job Parameters Guide:
-----------------------------------------
When starting a training job through SageMaker, the following parameters can be used to disable debuggers and source code packaging:
1. Main parameters:
   - disable_profiler=True  # Disable SageMaker profiler
   - debugger_hook_config=False  # Disable SageMaker debugger hooks
   - source_dir=None  # Don't use source directory, upload script directly

2. Environment variable settings (in the environment parameter):
   - SMDEBUG_DISABLED: 'true'
   - SM_DISABLE_DEBUGGER: 'true'
   - SAGEMAKER_DISABLE_PROFILER: 'true'
   - SMPROFILER_DISABLED: 'true'
   - SAGEMAKER_DISABLE_SOURCEDIR: 'true'

3. Example SageMaker code with disabled debugger:
```python
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()

estimator = PyTorch(
    entry_point='train_multi_model.py',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    disable_profiler=True,  # Disable profiler
    debugger_hook_config=False,  # Disable debugger hooks
    source_dir=None,  # Don't package source directory
    environment={
        'SMDEBUG_DISABLED': 'true',
        'SM_DISABLE_DEBUGGER': 'true',
        'SAGEMAKER_DISABLE_PROFILER': 'true',
        'SMPROFILER_DISABLED': 'true',
        'SAGEMAKER_DISABLE_SOURCEDIR': 'true',
    },
    hyperparameters={
        'task_name': 'TestTask',
        'epochs': 10,
    }
)

estimator.fit()
```
"""

# Import os module to ensure it's available for use in subsequent code
import os
import sys

# Disable SMDebug and Horovod to avoid PyTorch version conflicts
try:
    sys.modules['smdebug'] = None
    sys.modules['smddp'] = None
    sys.modules['smprofiler'] = None
    
    # Also disable Horovod
    sys.modules['horovod'] = None
    sys.modules['horovod.torch'] = None
    
    # Set all known environment variables to disable debugger
    os.environ['SMDEBUG_DISABLED'] = 'true'
    os.environ['SM_DISABLE_DEBUGGER'] = 'true'
    os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER'] = 'true'
    os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT'] = 'true'
    os.environ['SMPROFILER_DISABLED'] = 'true'
    os.environ['SM_SMDEBUG_DISABLED'] = 'true'
    os.environ['SM_SMDDP_DISABLE_PROFILING'] = 'true'
    os.environ['SAGEMAKER_DISABLE_PROFILER'] = 'true'
    
    # Disable source code wrapping and other features
    os.environ['SAGEMAKER_DISABLE_SOURCEDIR'] = 'true'
    os.environ['SAGEMAKER_CONTAINERS_IGNORE_SRC_REQUIREMENTS'] = 'true'
    os.environ['SAGEMAKER_DISABLE_BUILT_IN_PROFILER'] = 'true'
    os.environ['SAGEMAKER_DISABLE_DEFAULT_RULES'] = 'true'
    
    # Specifically disable file generation
    os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLED'] = 'true'
    # Disable the creation of training_job_end.ts file
    os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLE'] = 'true'
    # Disable debug-output directory
    os.environ['SAGEMAKER_DEBUG_OUTPUT_DISABLED'] = 'true'
    
    print("Successfully disabled SageMaker Debugger and related features")
except Exception as e:
    print(f"Warning: Error disabling modules: {e}")

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import math
import pandas as pd
import shutil

# Detect if running in SageMaker environment
is_sagemaker = 'SM_MODEL_DIR' in os.environ

# Custom collate function to handle None values in batch
def custom_collate_fn(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    # If all samples were None, return a dummy batch with empty tensors
    if len(batch) == 0:
        # Return a batch with empty tensors
        # This will let the dataloader continue instead of raising an error
        return torch.zeros(0, 1, 500, 232), torch.zeros(0, dtype=torch.long)
    
    # Use the default collate function for the filtered batch
    return torch.utils.data.dataloader.default_collate(batch)

# If running in SageMaker, import S3 tools
if is_sagemaker:
    import boto3
    s3_client = boto3.client('s3')
else:
    s3_client = None

# Print original command line arguments for diagnostic purposes
print("Original command line arguments:", sys.argv)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary models and data loaders
try:
from load.supervised.benchmark_loader import load_benchmark_supervised
    # Import model classes
    from model.supervised.models import (
        MLPClassifier, 
        LSTMClassifier, 
        ResNet18Classifier, 
        TransformerClassifier, 
        ViTClassifier,
        PatchTST,
        TimesFormer1D
    )
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

# Model factory dictionary
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier,
    'patchtst': PatchTST,
    'timesformer1d': TimesFormer1D
}

# Task trainer class (extracted from scripts/train_supervised.py)
from engine.supervised.task_trainer import TaskTrainer

def cleanup_sagemaker_storage():
    """
    Clean up unnecessary files in SageMaker environment to reduce storage usage
    """
    if not is_sagemaker:
        # Only run in SageMaker environment
        return
    
    logger.info("Cleaning up unnecessary files to reduce storage usage...")
    
    try:
        # First, ensure all debug-related environment variables are set
        os.environ['SMDEBUG_DISABLED'] = 'true'
        os.environ['SM_DISABLE_DEBUGGER'] = 'true'
        os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER'] = 'true'
        os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT'] = 'true'
        os.environ['SMPROFILER_DISABLED'] = 'true'
        os.environ['SM_SMDEBUG_DISABLED'] = 'true'
        os.environ['SM_SMDDP_DISABLE_PROFILING'] = 'true'
        os.environ['SAGEMAKER_DISABLE_PROFILER'] = 'true'
        os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLED'] = 'true'
        os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLE'] = 'true'
        os.environ['SAGEMAKER_DEBUG_OUTPUT_DISABLED'] = 'true'
        
        # Make sure debug-output directory exists before SageMaker tries to create it
        # This is a workaround - by creating it empty ourselves, we prevent SageMaker from creating and filling it
        os.makedirs("/opt/ml/output/debug-output", exist_ok=True)
        # Create an empty file to prevent SageMaker from creating the training_job_end.ts file
        with open("/opt/ml/output/debug-output/training_job_end.ts", "w") as f:
            f.write("")
        
        # Delete unnecessary temporary files and logs
        dirs_to_clean = [
            "/tmp",                        # Temporary directory
            "/opt/ml/output/profiler",     # Profiler output
            "/opt/ml/output/tensors",      # Debugger tensors
            "/opt/ml/output/debug-output", # Debug output directory (contains training_job_end.ts)
            "/opt/ml/code",                # Source code directory
            "/opt/ml/code/.sourcedir.tar.gz", # Source code package
            "/opt/ml/model",               # Model directory (not needed)
            "/opt/ml/output/intermediate", # Intermediate outputs
        ]
        
        # First check and delete specific problematic files
        problematic_files = [
            "/opt/ml/output/debug-output/training_job_end.ts",  # This file causes issues with output structure
            "/opt/ml/output/data/debug-output/training_job_end.ts", # Alternative location
            "/opt/ml/output/intermediate/training_job_end.ts",  # Another possible location
        ]
        
        for file_path in problematic_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed problematic file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not remove file {file_path}: {e}")
        
        # Clean up temporary directories
        for cleanup_dir in dirs_to_clean:
            if os.path.exists(cleanup_dir):
                logger.info(f"Cleaning directory: {cleanup_dir}")
                try:
                    # Only delete contents, not the directory itself
                    for item in os.listdir(cleanup_dir):
                        item_path = os.path.join(cleanup_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        elif os.path.isfile(item_path):
                            os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Error cleaning directory {cleanup_dir}: {e}")
        
        # Clean up sourcedir cache specifically
        sourcedir_cache = "/opt/ml/code/.sourcedir.tar.gz"
        if os.path.exists(sourcedir_cache):
            try:
                os.remove(sourcedir_cache)
                logger.info("Removed sourcedir cache")
            except Exception as e:
                logger.warning(f"Could not remove sourcedir cache: {e}")
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Clean up any manifest files that might have been generated
        for root, _, files in os.walk("/opt/ml/output/data"):
            for file in files:
                if (file.endswith('_uploads.txt') or 
                    file == 'upload_manifest.txt' or
                    file == 'training_job_end.ts'):
                    try:
                        os.remove(os.path.join(root, file))
                        logger.info(f"Removed manifest file: {file}")
                    except Exception as e:
                        logger.warning(f"Could not remove manifest file {file}: {e}")
        
        # Clean up duplicate output directories in task folder
        if os.path.exists("/opt/ml/output/data"):
            for task_dir in os.listdir("/opt/ml/output/data"):
                task_path = os.path.join("/opt/ml/output/data", task_dir)
                if os.path.isdir(task_path):
                    # Check for duplicated output subdirectories
                    output_dir = os.path.join(task_path, "output")
                    if os.path.exists(output_dir) and os.path.isdir(output_dir):
                        nested_output = os.path.join(output_dir, "output")
                        if os.path.exists(nested_output) and os.path.isdir(nested_output):
                            logger.info(f"Found nested output directory: {nested_output}")
                            
                            # Move contents to proper location
                            try:
                                for item in os.listdir(nested_output):
                                    src = os.path.join(nested_output, item)
                                    dst = os.path.join(task_path, item)
                                    if os.path.exists(dst):
                                        if os.path.isdir(src):
                                            shutil.rmtree(dst)
                                        else:
                                            os.remove(dst)
                                    shutil.move(src, task_path)
                                
                                # Remove empty directories
                                shutil.rmtree(nested_output, ignore_errors=True)
                                if not os.listdir(output_dir):
                                    shutil.rmtree(output_dir, ignore_errors=True)
                                
                                logger.info(f"Cleaned up nested output directories")
                            except Exception as e:
                                logger.warning(f"Error reorganizing nested output: {e}")
        
        # Also remove any debug-output directories in our task results
        for root, dirs, _ in os.walk("/opt/ml/output/data"):
            for dir_name in dirs:
                if dir_name == "debug-output":
                    debug_dir = os.path.join(root, dir_name)
                    try:
                        shutil.rmtree(debug_dir)
                        logger.info(f"Removed debug-output directory: {debug_dir}")
                    except Exception as e:
                        logger.warning(f"Could not remove debug-output directory {debug_dir}: {e}")
        
        # Make sure the main /opt/ml/output/debug-output directory is empty
        if os.path.exists("/opt/ml/output/debug-output"):
            for item in os.listdir("/opt/ml/output/debug-output"):
                item_path = os.path.join("/opt/ml/output/debug-output", item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    logger.warning(f"Could not remove {item_path}: {e}")
        
        logger.info("Storage cleanup completed!")
    except Exception as e:
        logger.error(f"Error during storage cleanup: {e}")
        import traceback
        logger.error(traceback.format_exc())

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WiFi Sensing Multi-Model Training")
    
    # Task parameters
    parser.add_argument('--task_name', type=str, required=True, help='Task name to train')
    
    # Data related parameters
    parser.add_argument('--data_root', type=str, default='/opt/ml/input/data/training', help='Data root directory')
    parser.add_argument('--tasks_dir', type=str, default='tasks', help='Tasks directory')
    parser.add_argument('--data_key', type=str, default='data', help='Data key')
    parser.add_argument('--file_format', type=str, default='h5', choices=['h5', 'npz', 'pt'], help='Data file format')
    parser.add_argument('--use_root_data_path', action='store_true', default=True, help='Use root directory as data path')
    parser.add_argument('--adaptive_path', action='store_true', default=True, help='Adaptively search path')
    parser.add_argument('--try_all_paths', action='store_true', default=True, help='Try all possible data paths')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./saved_models', help='Output directory')
    parser.add_argument('--save_to_s3', type=str, default=None, help='S3 path for saving results (s3://bucket/path)')
    
    # Model parameters
    parser.add_argument('--models', type=str, default='mlp,lstm,resnet18,transformer', help='Models to train, comma separated')
    parser.add_argument('--win_len', type=int, default=500, help='Window length')
    parser.add_argument('--feature_size', type=int, default=232, help='Feature size')
    parser.add_argument('--in_channels', type=int, default=1, help='Input channels')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=32, help='Batch size')  # Supports both formats
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    # Transformer/ViT specific parameters
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # PatchTST specific parameters
    parser.add_argument('--patch_len', type=int, default=16, help='Patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=8, help='Stride for PatchTST')
    
    # TimesFormer1D specific parameters
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size for TimesFormer1D')
    
    # Test parameters
    parser.add_argument('--test_splits', type=str, default='test_id,test_ood,test_cross_env', help='Test splits, comma separated')
    
    # Visualization parameters
    parser.add_argument('--save_plots', action='store_true', help='Save visualizations and plots')
    parser.add_argument('--save_confusion_matrix', action='store_true', help='Save confusion matrices')
    parser.add_argument('--save_learning_curves', action='store_true', help='Save learning curves')
    parser.add_argument('--save_predictions', action='store_true', help='Save model predictions')
    parser.add_argument('--save_model', action='store_true', help='Save trained model weights')
    parser.add_argument('--plot_dpi', type=int, default=150, help='DPI for saved plots')
    parser.add_argument('--plot_format', type=str, default='png', choices=['png', 'jpg', 'pdf', 'svg'], help='File format for plots')
    
    # S3 upload parameters
    parser.add_argument('--verify_uploads', action='store_true', help='Verify S3 uploads')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum retry attempts for S3 uploads')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Parse known arguments
    args, unknown = parser.parse_known_args()
    
    # Convert models list to list
    args.all_models = [m.strip() for m in args.models.split(',')]
    
    # Convert test splits to list
    if args.test_splits == 'all':
        args.test_splits = ['test_id', 'test_ood', 'test_cross_env']
    else:
        args.test_splits = [ts.strip() for ts in args.test_splits.split(',')]
    
    # Process legacy parameters from SM_HP environment variables
    for k, v in os.environ.items():
        # Check SM_HP_* format environment variables
        if k.startswith('SM_HP_'):
            # Convert parameter name
            param_name = k[6:].lower().replace('-', '_')
            
            # Handle learning rate alias (learning_rate vs lr)
            if param_name == 'learning_rate':
                param_name = 'lr'
            
            # Check if parameter exists
            if hasattr(args, param_name):
                # Convert value based on parameter type
                if isinstance(getattr(args, param_name), bool):
                    if v.lower() in ('true', 'yes', '1'):
                        setattr(args, param_name, True)
                    elif v.lower() in ('false', 'no', '0'):
                        setattr(args, param_name, False)
                elif isinstance(getattr(args, param_name), int):
                    try:
                        setattr(args, param_name, int(v))
                    except ValueError:
                        pass
                elif isinstance(getattr(args, param_name), float):
                    try:
                        setattr(args, param_name, float(v))
                    except ValueError:
                        pass
                else:
                    setattr(args, param_name, v)
                    
                # Special case - models list
                if param_name == 'models':
                    args.all_models = [m.strip() for m in v.split(',')]
                    
                # Special case - test splits
                if param_name == 'test_splits' and v != 'all':
                    args.test_splits = [ts.strip() for ts in v.split(',')]
        
        # Check S3 output path
        if k == 'SAGEMAKER_S3_OUTPUT' and args.save_to_s3 is None:
            args.save_to_s3 = v
            print(f"Setting S3 output path from environment variable: {v}")
    
    # Set correct output directory in SageMaker environment
    if is_sagemaker:
        args.output_dir = '/opt/ml/output/data'
        logger.info(f"Running in SageMaker environment, setting output_dir to {args.output_dir}")
        
        # In SageMaker, enable all visualization options by default
        args.save_plots = True
        args.save_confusion_matrix = True
        args.save_learning_curves = True
        args.save_model = True
        args.save_predictions = True
        args.verify_uploads = True
    
    # Create output directories if needed
    if is_sagemaker:
        os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)
    
    # Print parameters
    print("\n===== Parameters =====")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=====================\n")
    
    return args

def set_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model_name, data, args, device, is_distributed=False, local_rank=0):
    """Train a model of the specified type"""
    logger.info(f"===== Starting training for {model_name.upper()} model =====")
    
    # Unpack data
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # Get train and validation sets
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    if val_loader is None:
        logger.warning("No validation data found. Using training data for validation.")
        val_loader = train_loader
    
    # Create model
    logger.info(f"Creating {model_name.upper()} model...")
    ModelClass = MODEL_TYPES[model_name.lower()]
    
    # Create model-specific parameter sets
    base_kwargs = {
        'num_classes': num_classes
    }
    
    # Model-specific parameters
    if model_name.lower() == 'mlp':
        model_kwargs = {
            **base_kwargs,
            'win_len': args.win_len,
            'feature_size': args.feature_size
        }
    elif model_name.lower() == 'lstm':
        model_kwargs = {
            **base_kwargs,
            'feature_size': args.feature_size,
            'dropout': args.dropout
        }
    elif model_name.lower() == 'resnet18':
        model_kwargs = {
            **base_kwargs,
            'in_channels': args.in_channels,
            'feature_size': args.feature_size
        }
    elif model_name.lower() == 'transformer':
        model_kwargs = {
            **base_kwargs,
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'win_len': args.win_len,
            'dropout': args.dropout
        }
    elif model_name.lower() == 'vit':
        model_kwargs = {
            **base_kwargs,
            'in_channels': args.in_channels,
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'emb_dim': args.emb_dim,
            'dropout': args.dropout
        }
    elif model_name.lower() == 'patchtst':
        model_kwargs = {
            **base_kwargs,
            'in_channels': args.in_channels,
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'emb_dim': args.emb_dim,
            'dropout': args.dropout,
            'patch_len': args.patch_len,
            'stride': args.stride
        }
    elif model_name.lower() == 'timesformer1d':
        model_kwargs = {
            **base_kwargs,
            'in_channels': args.in_channels,
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'emb_dim': args.emb_dim,
            'dropout': args.dropout,
            'patch_size': args.patch_size
        }
    else:
        # Generic fallback parameters
        model_kwargs = {
            **base_kwargs,
            'in_channels': args.in_channels,
            'win_len': args.win_len,
            'feature_size': args.feature_size
        }
    
    # Create model instance
    model = ModelClass(**model_kwargs)
    model = model.to(device)
    
    logger.info(f"Model created: {model_name}")
    
    # Create experiment ID from timestamp and model name
    import hashlib
    timestamp = int(time.time())
    experiment_id = f"params_{hashlib.md5(f'{model_name}_{args.task_name}_{args.seed}_{timestamp}'.encode()).hexdigest()[:8]}"
    
    # Create directory structure that matches local pipeline
    # /output_dir/task_name/model_name/experiment_id/
    results_dir = os.path.join(args.output_dir, args.task_name, model_name, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Create config
    config = {
        'model': model_name,
        'task': args.task_name,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'seed': args.seed,
        'experiment_id': experiment_id
    }
    
    # Save configuration
    config_path = os.path.join(results_dir, f"{model_name}_{args.task_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler
    num_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def warmup_cosine_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # Get test loaders
    test_loaders = {k: v for k, v in loaders.items() if k.startswith('test')}
    if not test_loaders:
        logger.warning("No test splits found in the dataset. Check split names and dataset structure.")
    else:
        logger.info(f"Loaded {len(test_loaders)} test splits: {list(test_loaders.keys())}")
    
    # Create TaskTrainer
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=results_dir,  # Use results_dir for plots, metrics, etc.
        num_classes=num_classes,
        config=config,
        label_mapper=label_mapper,
        distributed=is_distributed,
        local_rank=local_rank if is_distributed else 0
    )
    
    # Train the model with early stopping
    trained_model, training_results = trainer.train()
    
    # Track best epoch
    best_epoch = training_results.get('best_epoch', args.epochs)
    
    # Save training history
    if 'training_dataframe' in training_results:
        history = training_results['training_dataframe']
        history_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_train_history.csv")
        history.to_csv(history_file, index=False)
        logger.info(f"Saved training history to {history_file}")
    
    # Store overall metrics
    overall_metrics = {}
    
    # Run evaluation on each test set
    for test_name, test_loader in test_loaders.items():
        logger.info(f"Evaluating on {test_name}...")
        test_loss, test_accuracy = trainer.evaluate(test_loader)
        
        # Calculate metrics
        try:
            test_f1, classification_report = trainer.calculate_metrics(test_loader)
            
            # Save classification report
            if isinstance(classification_report, pd.DataFrame):
                report_file = os.path.join(results_dir, f"classification_report_{test_name}.csv")
                classification_report.to_csv(report_file)
                logger.info(f"Saved classification report for {test_name} to {report_file}")
            else:
                logger.warning(f"Classification report for {test_name} is not a DataFrame, but type: {type(classification_report)}")
            
            # Generate and save confusion matrix
            try:
                confusion_matrix = trainer.plot_confusion_matrix(test_loader, save_path=os.path.join(results_dir, f"{model_name}_{args.task_name}_{test_name}_confusion.png"))
            except Exception as e:
                logger.error(f"Error generating confusion matrix for {test_name}: {e}")
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics for {test_name}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())  # Print full stack trace
            test_f1 = 0.0
        
        # Store metrics
        overall_metrics[test_name] = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'f1_score': test_f1
        }
        
        logger.info(f"{test_name} Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # Generate and save learning curves
    try:
        if 'train_loss_history' in training_results and 'val_loss_history' in training_results:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(training_results['train_loss_history'])+1), training_results['train_loss_history'], label='Train Loss')
            plt.plot(range(1, len(training_results['val_loss_history'])+1), training_results['val_loss_history'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} on {args.task_name} - Loss Curves')
            plt.legend()
            plt.grid(True)
            learning_curve_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_learning_curves.png")
            plt.savefig(learning_curve_file)
            plt.close()
            logger.info(f"Saved learning curves to {learning_curve_file}")
            
            # Accuracy curves
            if 'train_accuracy_history' in training_results and 'val_accuracy_history' in training_results:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(training_results['train_accuracy_history'])+1), training_results['train_accuracy_history'], label='Train Accuracy')
                plt.plot(range(1, len(training_results['val_accuracy_history'])+1), training_results['val_accuracy_history'], label='Val Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title(f'{model_name} on {args.task_name} - Accuracy Curves')
                plt.legend()
                plt.grid(True)
                accuracy_curve_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_accuracy_curves.png")
                plt.savefig(accuracy_curve_file)
                plt.close()
                logger.info(f"Saved accuracy curves to {accuracy_curve_file}")
    except Exception as e:
        logger.error(f"Error generating learning curves: {e}")
    
    # Save test results
    results_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_results.json")
    
    # Make sure results are JSON serializable
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj
    
    # Process all metrics
    serializable_metrics = {}
    for key, value in overall_metrics.items():
        serializable_metrics[key] = {k: convert_to_json_serializable(v) for k, v in value.items()}
    
    with open(results_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    # Save summary
    try:
        summary = {
            'best_epoch': best_epoch,
            'experiment_id': experiment_id,
            'experiment_completed': True
        }
        
        # Add val metrics if available
        if 'val_accuracy_history' in training_results and len(training_results['val_accuracy_history']) > 0:
            best_idx = best_epoch - 1 if best_epoch > 0 else 0
            if best_idx < len(training_results['val_accuracy_history']):
                summary['best_val_accuracy'] = training_results['val_accuracy_history'][best_idx]
                
            if 'val_loss_history' in training_results and best_idx < len(training_results['val_loss_history']):
                summary['best_val_loss'] = training_results['val_loss_history'][best_idx]
        
        # Add test results to summary
        for split_name, metrics in serializable_metrics.items():
            summary[f'{split_name}_accuracy'] = metrics['accuracy']
            summary[f'{split_name}_f1_score'] = metrics['f1_score']
        
        summary_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Update best performance tracking
        model_dir = os.path.dirname(results_dir)
        best_performance_file = os.path.join(model_dir, "best_performance.json")
        
        # Check if there's an existing best performance file
        best_performance = {}
        if os.path.exists(best_performance_file):
            try:
                with open(best_performance_file, 'r') as f:
                    best_performance = json.load(f)
            except:
                pass
        
        # Calculate average test accuracy to determine if this is the best run
        avg_test_accuracy = sum([metrics['accuracy'] for metrics in serializable_metrics.values()]) / len(serializable_metrics)
        
        # Update best performance if this run is better
        if not best_performance or avg_test_accuracy > best_performance.get('avg_test_accuracy', 0):
            best_performance = {
                'experiment_id': experiment_id,
                'avg_test_accuracy': avg_test_accuracy,
                'best_epoch': best_epoch,
                'timestamp': time.time(),
                'test_metrics': serializable_metrics,
                'experiment_path': results_dir
            }
            
            with open(best_performance_file, 'w') as f:
                json.dump(best_performance, f, indent=4)
                
            logger.info(f"Updated best performance record for {model_name} on {args.task_name}")
            
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    # Save overall test accuracy
    if len(serializable_metrics) > 0:
        all_test_accuracies = [metrics.get('accuracy', 0) for metrics in serializable_metrics.values()]
        overall_metrics['test_accuracy'] = sum(all_test_accuracies) / len(all_test_accuracies)
    
    logger.info(f"Training and evaluation completed. Results saved to {results_dir}")
    
    return trained_model, overall_metrics

def main():
    """
    Main function - Train multiple models on a specified task
    """
    try:
        # Disable SageMaker debugger and profiler
        os.environ['SMDEBUG_DISABLED'] = 'true'
        os.environ['SM_DISABLE_DEBUGGER'] = 'true'
        os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER'] = 'true'
        os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT'] = 'true'
        os.environ['SMPROFILER_DISABLED'] = 'true'
        os.environ['SAGEMAKER_DISABLE_SOURCEDIR'] = 'true'  # Disable sourcedir packaging
        os.environ['SAGEMAKER_CONTAINERS_IGNORE_SRC_REQUIREMENTS'] = 'true'
        os.environ['SAGEMAKER_DISABLE_BUILT_IN_PROFILER'] = 'true'
        os.environ['SAGEMAKER_DISABLE_DEFAULT_RULES'] = 'true'
        os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLED'] = 'true'
        
        # Set GPU optimization parameters
        if torch.cuda.is_available():
            # Enable cuDNN auto-tuning
            torch.backends.cudnn.benchmark = True
            # For fixed-size inputs, using cuDNN deterministic mode can improve performance
            torch.backends.cudnn.deterministic = False
            # Configure GPU memory allocator optimizations
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            # Clear GPU cache
            torch.cuda.empty_cache()
            # Display available GPU count
            logger.info(f"Available GPU count: {torch.cuda.device_count()}")
        
        # Check distributed training environment
        is_distributed = False
        local_rank = 0
        world_size = 1
        rank = 0
        
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            is_distributed = True
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            world_size = int(os.environ.get('WORLD_SIZE', '1'))
            rank = int(os.environ.get('RANK', '0'))
            
            logger.info(f"Running in distributed environment: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, RANK={rank}")
            
            # Confirm distributed initialization is complete
            if torch.distributed.is_initialized():
                logger.info("Distributed backend already initialized")
            else:
                logger.info("Distributed backend not yet initialized, attempting to initialize now")
                try:
                    # Try to initialize distributed environment (if not already done in entry_script.py)
                    torch.distributed.init_process_group(backend='nccl')
                    torch.cuda.set_device(local_rank)
                    logger.info("Successfully initialized distributed backend")
                except Exception as e:
                    logger.warning(f"Unable to initialize distributed backend: {e}")
        else:
            logger.info("Running in single GPU environment")
        
        # Log environment variables for debugging
        if is_sagemaker:
            logger.info("Running in SageMaker environment")
            logger.info("Environment variables:")
            for key in sorted([k for k in os.environ.keys() if k.startswith(('SM_', 'SAGEMAKER_'))]):
                logger.info(f"  {key}: {os.environ.get(key)}")

        # Get command line arguments
        args = get_args()
        
        # Log parsed arguments
        logger.info("Parsed arguments:")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info(f"  {arg_name}: {arg_value}")
        
        # Set device
        if is_distributed:
            # In distributed environment, set device based on local_rank
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Process {rank} using device: {device} (local_rank: {local_rank})")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
        
        # Configure thread optimizations
        if device.type == 'cuda':
            # Limit CPU thread count to reduce thread contention
            torch.set_num_threads(4)  # Set PyTorch thread count
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(4)  # Set inter-op parallelism threads
            
            # Try to enable TensorFloat32 precision mode (for Ampere or newer GPUs)
            if hasattr(torch.cuda, 'matmul') and hasattr(torch.cuda.matmul, 'allow_tf32'):
                torch.cuda.matmul.allow_tf32 = True  # Allow TF32 in matrix multiplications
                logger.info("Enabled TensorFloat32 precision mode to accelerate matrix operations")
            
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 in cuDNN operations
                logger.info("Enabled cuDNN TensorFloat32 precision mode")
                
            # Display memory usage for each GPU
            for i in range(torch.cuda.device_count()):
                gpu_total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"GPU #{i} ({torch.cuda.get_device_name(i)}): "
                           f"Total memory={gpu_total_mem:.2f}GB, "
                           f"Allocated={gpu_allocated:.2f}GB, "
                           f"Cached={gpu_reserved:.2f}GB")
        
        # Set random seed
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Log training start
        logger.info(f"Starting multi-model training, task: {args.task_name}")
        logger.info(f"Models to train: {args.all_models}")
        
        # Check for S3 output path in environment variables if not specified
        if args.save_to_s3 is None and is_sagemaker:
            sm_output_s3 = os.environ.get('SAGEMAKER_S3_OUTPUT')
            if sm_output_s3:
                logger.info(f"Found S3 output path in environment variables: {sm_output_s3}")
                args.save_to_s3 = sm_output_s3
        
        if args.save_to_s3:
            logger.info(f"Results will be uploaded to S3: {args.save_to_s3}")
        
        # Set data root directory - directly use SM_CHANNEL_TRAINING in SageMaker
        dataset_root = '/opt/ml/input/data/training' if is_sagemaker else args.data_root
        logger.info(f"Using data root directory: {dataset_root}")
        
        # Check for actual H5 file keys if running in SageMaker
        if is_sagemaker and args.file_format == 'h5':
            try:
                import h5py
                # Find an H5 file to inspect
                h5_files = []
                for root, _, files in os.walk(dataset_root):
                    for file in files:
                        if file.endswith('.h5'):
                            h5_files.append(os.path.join(root, file))
                            break
                    if h5_files:
                        break
                
                if h5_files:
                    # Open first H5 file and check available keys
                    with h5py.File(h5_files[0], 'r') as f:
                        available_keys = list(f.keys())
                        logger.info(f"H5 file contains these keys: {available_keys}")
                        
                        # If 'data' key is not found, try to use the first available key
                        if 'data' not in available_keys and available_keys:
                            # Check specifically for 'CSI_amps' which is common in WiFi datasets
                            if 'CSI_amps' in available_keys:
                                args.data_key = 'CSI_amps'
                            else:
                                args.data_key = available_keys[0]
                            logger.info(f"Setting data_key to '{args.data_key}' based on file contents")
            except Exception as e:
                logger.warning(f"Failed to inspect H5 files: {e}")
        
        # Adjust batch size for distributed training
        if is_distributed and args.batch_size % world_size != 0:
            old_batch_size = args.batch_size
            args.batch_size = (args.batch_size // world_size) * world_size
            if args.batch_size == 0:
                args.batch_size = world_size
            logger.info(f"Adjusting batch size for distributed training: {old_batch_size} -> {args.batch_size}")
        
        # Load data
        logger.info(f"Loading data from {dataset_root}, task name: {args.task_name}")
        logger.info(f"Using batch size: {args.batch_size}")
        data = load_benchmark_supervised(
            dataset_root=dataset_root,
            task_name=args.task_name,
            batch_size=args.batch_size,
            data_key=args.data_key,
            file_format=args.file_format,
            num_workers=args.num_workers,
            use_root_as_task_dir=args.use_root_data_path,
            debug=False,  # Disable debug print messages
            distributed=is_distributed,  # Add distributed training flag
            collate_fn=custom_collate_fn  # Add custom collate function
        )
        
        # Check if data loaded successfully
        if not data or 'loaders' not in data:
            logger.error(f"Failed to load data for task {args.task_name}")
            sys.exit(1)
        
        logger.info(f"Data loaded successfully. Number of classes: {data['num_classes']}")
        logger.info(f"Available data loaders: {list(data['loaders'].keys())}")
        
        # Verify batch size in data loader
        if 'train' in data['loaders']:
            batch_size_used = next(iter(data['loaders']['train']))[0].shape[0]
            logger.info(f"Actual batch size used in train loader: {batch_size_used}")
            if batch_size_used != args.batch_size and not is_distributed:
                logger.warning(f"WARNING: Actual batch size {batch_size_used} differs from requested batch size {args.batch_size}")
        
        # Track model results
        successful_models = []
        failed_models = []
        
        # Store all results
        all_results = {}
        
        # Train each model
        for model_name in args.all_models:
            try:
                logger.info(f"\n{'='*40}\nTraining model: {model_name}\n{'='*40}")
                
                # Verify model compatibility
                try:
                    ModelClass = MODEL_TYPES[model_name.lower()]
                    logger.info(f"Model class {model_name} loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model class {model_name}: {e}")
                    failed_models.append((model_name, f"Model class error: {str(e)}"))
                    continue
                
                # Train model
                model, metrics = train_model(model_name, data, args, device, is_distributed=is_distributed, local_rank=local_rank)
                
                # Check if training succeeded
                if model is None or (isinstance(metrics, dict) and 'error' in metrics):
                    error_msg = metrics.get('error', 'Unknown error') if isinstance(metrics, dict) else 'Unknown error'
                    logger.error(f"Model {model_name} training failed: {error_msg}")
                    failed_models.append((model_name, error_msg))
                else:
                    all_results[model_name] = metrics
                    successful_models.append(model_name)
                    logger.info(f"Completed training for {model_name}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed_models.append((model_name, str(e)))
        
        # Print training summary
        logger.info("\n" + "="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        logger.info(f"Task: {args.task_name}")
        logger.info(f"Successfully trained models ({len(successful_models)}): {', '.join(successful_models)}")
        logger.info(f"Failed models ({len(failed_models)}): {', '.join([m[0] for m in failed_models])}")
        
        if failed_models:
            logger.info("\nFailure details:")
            for model_name, error in failed_models:
                logger.info(f"  - {model_name}: {error}")
        
        # Save overall results summary
        results_path = os.path.join(args.output_dir, args.task_name, "multi_model_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"All training completed. Results saved to {results_path}")
        logger.info("Results summary:")
        for model_name, metrics in all_results.items():
            logger.info(f"  - {model_name}: Test accuracy = {metrics.get('test_accuracy', 0.0):.4f}")
        
        # Upload results to S3 if running in SageMaker
        if is_sagemaker and args.save_to_s3:
            logger.info(f"Uploading results to S3: {args.save_to_s3}")
            
            # Only upload the specific task directory to maintain clean structure
            task_dir = os.path.join(args.output_dir, args.task_name)
            
            if os.path.exists(task_dir):
                # Construct S3 destination path without job timestamp
                s3_base_path = args.save_to_s3.rstrip('/')
                
                # Extract the bucket name and prefix without job timestamp
                parts = s3_base_path.replace('s3://', '').split('/')
                bucket_name = parts[0]
                
                # Remove any timestamp or job ID from the path to avoid nested directories
                # Examples to clean:
                # - s3://bucket/Benchmark_Log/TaskName-20240510-1213/ -> s3://bucket/Benchmark_Log/
                # - s3://bucket/Benchmark_Log/TestTask-20250510-1608/output/output/ -> s3://bucket/Benchmark_Log/
                prefix_parts = []
                for part in parts[1:]:
                    # Skip parts that look like timestamps or job IDs (contain numbers and dashes)
                    if not (any(c.isdigit() for c in part) and '-' in part):
                        # Also skip duplicated 'output' directories
                        if part != 'output' or 'output' not in prefix_parts:
                            prefix_parts.append(part)
                
                # Create clean base path
                clean_prefix = '/'.join(prefix_parts)
                
                # Final S3 path: s3://bucket/Benchmark_Log/task_name/
                task_s3_path = f"s3://{bucket_name}/{clean_prefix}/{args.task_name}"
                
                logger.info(f"Uploading results to clean S3 path: {task_s3_path}")
                
                # Upload the task directory directly to avoid nested output folders
                try:
                    # Simple upload with retries
                    upload_success = False
                    max_retries = 3
                    
                    for retry in range(max_retries):
                        try:
                            logger.info(f"Uploading task directory (attempt {retry+1}/{max_retries}): {task_dir} -> {task_s3_path}")
                            
                            # Count files for progress reporting
                            total_files = sum([len(files) for _, _, files in os.walk(task_dir)])
                            logger.info(f"Found {total_files} files to upload")
                            
                            # Track successful uploads
                            uploaded_count = 0
                            
                            # Walk and upload each file
                            for root, dirs, files in os.walk(task_dir):
                                # Skip debug-output directories and similar
                                if any(skip_dir in root for skip_dir in ['debug-output', 'profiler', 'tensors']):
                                    continue
                                
                                for file in files:
                                    local_file_path = os.path.join(root, file)
                                    
                                    # Skip problematic files by name or path
                                    if (file == 'training_job_end.ts' or 
                                        'debug-output' in local_file_path or 
                                        'profiler' in local_file_path or
                                        file.endswith('_uploads.txt') or 
                                        file == 'upload_manifest.txt'):
                                        continue
                                    
                                    # Calculate the relative path to maintain directory structure
                                    rel_path = os.path.relpath(local_file_path, task_dir)
                                    
                                    # Clean relative path - avoid duplicated directories
                                    rel_parts = rel_path.split(os.sep)
                                    clean_rel_parts = []
                                    for part in rel_parts:
                                        # Skip duplicate 'output' directories
                                        if part != 'output' or 'output' not in clean_rel_parts:
                                            clean_rel_parts.append(part)
                                    
                                    clean_rel_path = os.path.join(*clean_rel_parts) if clean_rel_parts else rel_path
                                    
                                    # Final S3 key
                                    s3_key = f"{clean_prefix}/{args.task_name}/{clean_rel_path}"
                                    
                                    # Upload file
                                    s3_client.upload_file(local_file_path, bucket_name, s3_key)
                                    uploaded_count += 1
                                    
                                    # Log progress periodically
                                    if uploaded_count % 10 == 0 or uploaded_count == total_files:
                                        logger.info(f"Upload progress: {uploaded_count}/{total_files} files ({uploaded_count/total_files*100:.1f}%)")
                        
                            logger.info(f"Successfully uploaded {uploaded_count} files to {task_s3_path}")
                            upload_success = True
                            break
                        except Exception as e:
                            logger.warning(f"Upload attempt {retry+1} failed: {e}")
                            if retry < max_retries - 1:
                                logger.info(f"Retrying in 5 seconds...")
                                time.sleep(5)
                    
                    if not upload_success:
                        logger.error(f"Failed to upload results after {max_retries} attempts")
                except Exception as e:
                    logger.error(f"Error during upload: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.error(f"Task directory {task_dir} does not exist. Cannot upload results.")
        
        # Clean up SageMaker storage
        if is_sagemaker:
            cleanup_sagemaker_storage()
        
        logger.info("Multi-model training completed successfully!")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 
