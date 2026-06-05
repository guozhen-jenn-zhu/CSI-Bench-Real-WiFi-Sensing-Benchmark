#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script enables running supervised learning and multi-task learning pipelines in a SageMaker environment.
It creates a SageMaker PyTorch Estimator to submit training jobs.

Main features:
1. Batch execution of training jobs, with each job running multiple models on a single instance
2. Support for multi-task learning and few-shot learning
3. Support for configuring parameters from JSON files

Usage example:
```
python sagemaker_runner.py --config configs/my_custom_config.json
```
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3

# Default path settings
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Root directory (one level up from scripts)
CONFIG_DIR = os.path.join(CODE_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "sagemaker_default_config.json")

def load_config(config_path=None):
    """Load configuration from JSON file"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Process few-shot configuration to ensure backward compatibility
        if 'fewshot' in config:
            fewshot_config = config['fewshot']
            # Set compatibility parameters
            config['enable_few_shot'] = fewshot_config.get('enabled', False)
            config['k_shot'] = fewshot_config.get('k_shots', 5)
            config['inner_lr'] = fewshot_config.get('adaptation_lr', 0.01)
            config['num_inner_steps'] = fewshot_config.get('adaptation_steps', 10)
            config['fewshot_support_split'] = fewshot_config.get('support_split', 'val_id')
            config['fewshot_query_split'] = fewshot_config.get('query_split', 'test_cross_env')
            config['fewshot_finetune_all'] = fewshot_config.get('finetune_all', False)
            config['fewshot_eval_shots'] = fewshot_config.get('eval_shots', False)
            
        print(f"Configuration loaded from {config_path}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Unable to load configuration file: {e}")
        sys.exit(1)

# Load default configuration
DEFAULT_CONFIG = load_config()

class SageMakerRunner:
    """Class that handles SageMaker training job creation and execution"""
    
    def __init__(self, config):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # Shorter format for job names
        
        # Load configuration
        self.config = config
        
        # Extract common parameters from configuration
        self.s3_data_base = self.config.get("s3_data_base")
        self.s3_output_base = self.config.get("s3_output_base")
        
        # Configure SageMaker training job default params for this runner
        boto3_session = boto3.Session()
        self.sm_client = boto3_session.client('sagemaker')
        
        # Disable debug outputs globally if possible
        self._configure_disable_debug_outputs()
        
        # Verify S3 bucket exists
        self._verify_s3_bucket()
        
        print(f"SageMaker Runner initialized:")
        print(f"  S3 data base path: {self.s3_data_base}")
        print(f"  S3 output base path: {self.s3_output_base}")
        print(f"  Timestamp: {self.timestamp}")
    
    def _configure_disable_debug_outputs(self):
        """Try to disable debug outputs globally at the SageMaker client level if possible"""
        try:
            # Some debug output control might be enabled via SageMaker client configuration
            # or via AWS CLI commands. This method tries to apply those if available.
            
            # Check if boto3 client supports configuration of these settings
            client_config = self.sm_client._get_config_value('training_job_preferences', {})
            
            # Try to update client config if method exists
            # Note: This is AWS internal and might not be supported, just a placeholder
            # for future SageMaker improvements
            if hasattr(self.sm_client, '_modify_config'):
                training_prefs = {
                    'debug_output_enabled': False,
                    'profiler_enabled': False,
                    'auto_upload_enabled': False
                }
                try:
                    self.sm_client._modify_config('training_job_preferences', training_prefs)
                    print("Successfully configured global SageMaker client training preferences")
                except:
                    pass
            
            # Alternative option - use AWS CLI to set preferences
            # This is a speculative approach and might not work on all environments
            try:
                import subprocess
                subprocess.run(
                    ["aws", "configure", "set", "sagemaker.disable_debugger", "true"], 
                    check=True, capture_output=True
                )
                print("Set AWS CLI sagemaker.disable_debugger config")
            except:
                # Silently ignore errors - this is just a best-effort attempt
                pass
        except Exception as e:
            print(f"Note: Could not configure global SageMaker options: {e}")
            # We'll fall back to per-job configuration
    
    def _verify_s3_bucket(self):
        """Verify S3 bucket exists and list available data"""
        try:
            s3 = boto3.resource('s3')
            bucket_name = self.s3_data_base.split('/')[2]  # Extract bucket name from S3 path
            
            # Check if the bucket exists
            if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
                print(f"Error: Bucket '{bucket_name}' does not exist. Please create it first.")
                sys.exit(1)
        
            # Check contents of the S3 path
            s3_client = boto3.client('s3')
            bucket = self.s3_data_base.split('/')[2]
            prefix = '/'.join(self.s3_data_base.split('/')[3:])
            if not prefix.endswith('/'):
                prefix += '/'
        
            # Try to list contents of the S3 path
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            
            if 'CommonPrefixes' in response:
                print(f"Contents of S3 path {self.s3_data_base}:")
                for obj in response['CommonPrefixes']:
                    folder = obj['Prefix'].split('/')[-2]
                    print(f"  - {folder}/")

                # New ``CSI-Bench`` layout puts tasks directly under the data
                # base; the legacy layout used a ``tasks/`` subdir.  Try the
                # new layout first, then fall back to the legacy one.
                tried_legacy = False
                for sub_prefix in ('', 'tasks/', 'Multitask/'):
                    full_prefix = prefix + sub_prefix
                    subresp = s3_client.list_objects_v2(
                        Bucket=bucket, Prefix=full_prefix, Delimiter='/'
                    )
                    if 'CommonPrefixes' in subresp:
                        label = full_prefix
                        print(f"Available tasks in s3://{bucket}/{label}:")
                        for obj in subresp['CommonPrefixes']:
                            task_name = obj['Prefix'].split('/')[-2]
                            print(f"  - {task_name}/")
                    elif sub_prefix == 'tasks/':
                        tried_legacy = True
                if tried_legacy:
                    print(
                        f"(No legacy 'tasks/' subdir under {self.s3_data_base}; "
                        f"that's fine for the new CSI-Bench layout.)"
                    )
            else:
                print(f"Warning: S3 path {self.s3_data_base} appears to be empty")
        except Exception as e:
            print(f"Warning: Error checking S3 path {self.s3_data_base}: {e}")
    
    def run_batch_by_task(self, tasks=None, models=None, override_config=None):
        """
        Run batch by task, executing each task on a single instance.
        
        Parameters:
            tasks (list): List of tasks to process (defaults to all available tasks)
            models (list): List of models to run for each task (defaults to all available models)
            override_config (dict): Configuration override parameters
            
        Returns:
            list: List of job information dictionaries
        """
        print(f"Starting batch execution by task...")
        
        # Create batch timestamp for job naming
        batch_timestamp = self.timestamp
        
        # Set defaults if not provided
        tasks_to_run = tasks or self.config.get('available_tasks')
        # Always use all available models from config by default
        models_to_run = self.config.get('available_models')
        
        # Convert to lists if strings are provided
        if isinstance(tasks_to_run, str):
            tasks_to_run = [tasks_to_run]
        if isinstance(models_to_run, str):
            models_to_run = [models_to_run]
        
        print(f"Will train all available models for each task: {models_to_run}")
        
        # Prepare configuration
        base_config = dict(self.config)
        if override_config:
            base_config.update(override_config)
        
        # Get instance types for tasks (if specified as a list)
        instance_types = base_config.get("instance_type", [])
        if not isinstance(instance_types, list):
            # If instance_type is not a list, convert it to a single-item list
            instance_types = [instance_types]
        
        # List to track job information
        jobs = []
        
        # Iterate through tasks
        for i, task in enumerate(tasks_to_run):
            print(f"\nProcessing task: {task}")
            
            # Create task-specific configuration
            task_config = dict(base_config)
            task_config['task'] = task
            
            # Set instance type for this task
            # If instance_types list is shorter than tasks_to_run, use the last available instance type
            if instance_types:
                if i < len(instance_types):
                    task_instance_type = instance_types[i]
                else:
                    task_instance_type = instance_types[-1]
                
                task_config['instance_type'] = task_instance_type
                print(f"Using instance type '{task_instance_type}' for task '{task}'")
            
            # Add use_root_data_path parameter for the task
            # Supports the following path lookup strategies:
            # 1. Directly specify use_root_data_path = True/False
            # 2. Use different path strategies for different tasks (root_data_tasks config item)
            # 3. Try different paths for first task, then use successful path for subsequent tasks
            
            # First check if this task is in the root_data_tasks list
            root_data_tasks = task_config.get('root_data_tasks', [])
            if isinstance(root_data_tasks, str):
                root_data_tasks = [t.strip() for t in root_data_tasks.split(',')]
                
            if task in root_data_tasks:
                print(f"Task '{task}' is configured to use root data path")
                task_config['use_root_data_path'] = True
            
            # Create an estimator that will run all models for this task
            estimator = self._create_estimator(
                task_config,
                base_job_name=f"{task_config.get('base_job_name', 'wifi-sensing')}-{task}",
                models=models_to_run
            )
            
            # Create job name and replace underscores with dashes to avoid SageMaker naming restrictions
            # Keep the task name unchanged for S3 paths but modify it for job naming
            job_name = f"{task.replace('_', '-')}-{batch_timestamp}"
            
            # Launch training job
            estimator.fit(
                inputs=self._prepare_inputs(task_config),
                job_name=job_name,
                wait=False,
                logs=False  # Don't stream logs automatically
            )
            
            # Track job information
            jobs.append({
                'job_name': job_name,
                'task': task,
                'models': models_to_run,
                'status': 'InProgress',
                'estimator': estimator,
                'instance_type': task_config.get('instance_type')
            })
            
            print(f"Job submitted: {job_name} (instance: {task_config.get('instance_type')})")
            
            # If waiting time is specified, wait between jobs
            if len(tasks_to_run) > 1 and task != tasks_to_run[-1]:
                wait_time = task_config.get('batch_wait_time', 30)
                print(f"Waiting {wait_time} seconds before submitting next job...")
                time.sleep(wait_time)
        
        return jobs
    
    def _create_estimator(self, config, base_job_name, models=None):
        """
        Create SageMaker PyTorch Estimator from given configuration
        """
        # Always use all available models from config
        all_models = config.get('available_models', [])
        if all_models:
            print(f"Using all available models: {all_models}")
        else:
            print("Warning: No models defined in 'available_models' config")
            all_models = ['mlp', 'lstm', 'resnet18', 'transformer']
            print(f"Using default models: {all_models}")
            
        # Get instance type (may be a single value or from a list per task)
        instance_type = config.get('instance_type', 'ml.g4dn.xlarge')
        # Ensure instance_type is a string, not a list
        if isinstance(instance_type, list):
            instance_type = instance_type[0]
            print(f"Converting instance_type from list to single value: {instance_type}")
            
        print(f"Using instance type: {instance_type}")
            
        # Prepare hyperparameters - Use dashes instead of underscores for parameter names
        hyperparameters = {
            'models': ','.join(all_models),
            'task_name': config.get('task_name', config.get('task')),  # Keep 'task_name' as is - this is critical
            'win-len': config.get('win_len', 500),  # Match default value from train_supervised.py
            'feature-size': config.get('feature_size', 232),
            'batch-size': config.get('batch_size', 32),  # Match default value from train_multi_model.py
            'epochs': config.get('epochs', 100),
            'test-splits': config.get('test_splits', 'all'),
            'seed': config.get('seed', 42),
            'learning-rate': config.get('learning_rate', 0.001),  # Add default learning rate
            'weight-decay': config.get('weight_decay', 1e-5),  # Add default weight decay
            'warmup-epochs': config.get('warmup_epochs', 5),  # Add default warmup period
            'patience': config.get('patience', 15),  # Add default patience value
            'adaptive-path': "True" if config.get('adaptive_path', False) else "False",  # Add adaptive path option
            'try-all-paths': "True" if config.get('try_all_paths', False) else "False",  # Add try all paths option
            'use-root-data-path': "True",  # Always use root directory data by default
            'direct-upload': "True",  # Force direct upload to S3
            'save-to-s3': self.s3_output_base,  # Set S3 output path for saving
            'd-model': config.get('d_model', 64),  # Transformer model dimension
            'emb-dim': config.get('emb_dim', 64),  # Embedding dimension
            'dropout': config.get('dropout', 0.1),  # Dropout rate
            'in-channels': config.get('in_channels', 1),  # Input channels
            'patch-len': config.get('patch_len', 16),  # PatchTST patch length
            'stride': config.get('stride', 8),  # PatchTST stride
            'patch-size': config.get('patch_size', 4)  # TimesFormer1D patch size
        }
        
        # Check and log batch size parameter
        orig_batch_size = config.get('batch_size', 32)
        print(f"Original batch_size from config: {orig_batch_size}")
        print(f"Passed batch-size parameter: {hyperparameters['batch-size']}")
        
        # Ensure batch size is correctly passed
        if orig_batch_size != hyperparameters['batch-size']:
            print(f"WARNING: Original batch_size {orig_batch_size} differs from passed batch-size {hyperparameters['batch-size']}")
            # Force use of batch size from config file
            hyperparameters['batch-size'] = orig_batch_size
            print(f"Updated batch-size to match config: {hyperparameters['batch-size']}")
        
        # As a redundancy measure, also add underscore version of batch size parameter
        hyperparameters['batch_size'] = orig_batch_size
        
        # Add few-shot parameters (if enabled)
        if config.get('enable_few_shot', False) or config.get('fewshot_eval_shots', False):
            hyperparameters['enable-few-shot'] = "True"
            hyperparameters['k-shot'] = config.get('k_shot', 5)
            hyperparameters['inner-lr'] = config.get('inner_lr', 0.01)
            hyperparameters['num-inner-steps'] = config.get('num_inner_steps', 10)
            
            # Add few-shot support and query splits (if they exist)
            for param in ['fewshot_support_split', 'fewshot_query_split']:
                if param in config:
                    # Replace underscores with dashes in parameter name
                    fixed_param = param.replace('_', '-')
                    hyperparameters[fixed_param] = config[param]
        
        # Add model-specific parameters (if they exist)
        if 'model_params' in config:
            for key, value in config['model_params'].items():
                # Replace underscores with dashes in parameter name
                fixed_key = key.replace('_', '-')
                hyperparameters[fixed_key] = value
                
        # Print hyperparameters for debugging
        print(f"Setting up hyperparameters for estimator:")
        for key, value in sorted(hyperparameters.items()):
            print(f"  {key}: {value}")
            
        # Special focus on S3 output path
        print(f"S3 output path for results: {self.s3_output_base}")
        
        # Create estimator
        estimator = PyTorch(
            entry_point='scripts/entry_script.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=config.get('framework_version', '1.12.1'),
            py_version=config.get('py_version', 'py38'),
            instance_count=config.get('instance_count', 1),
            instance_type=instance_type,
            base_job_name=base_job_name,
            hyperparameters=hyperparameters,
            max_run=config.get('max_run', 24 * 3600),  # Default 24-hour maximum run time
            keep_alive_period_in_seconds=config.get('keep_alive_period', 1200),  # Default keep instance active 20 minutes
            output_path=self.s3_output_base,  # Explicitly set output path
            code_location=self.s3_output_base,  # Where to store the code package
            debugger_hook_config=False,  # Disable debugger hooks completely
            disable_upload_notifications=True,  # Disable notifications about uploads
            profiler_config=None,  # No profiler config
            # Enable distributed training functionality
            distribution={
                'pytorch': {
                    # Use PyTorch's DistributedDataParallel (DDP)
                    'enabled': True,  
                    # Use NCCL backend for inter-GPU communication
                    'backend': 'nccl'
                }
            },
            environment={
                'SAGEMAKER_S3_OUTPUT': self.s3_output_base,  # Set environment variable for S3 output path
                'SMDEBUG_DISABLED': 'true',
                'SM_DISABLE_DEBUGGER': 'true',
                'SMDATAPARALLEL_DISABLE_DEBUGGER': 'true',
                'SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT': 'true',
                'SMPROFILER_DISABLED': 'true',
                'SM_SMDEBUG_DISABLED': 'true',
                'SM_SMDDP_DISABLE_PROFILING': 'true',
                'SAGEMAKER_DISABLE_PROFILER': 'true',
                'SAGEMAKER_DISABLE_SOURCEDIR': 'true',
                'SAGEMAKER_CONTAINERS_IGNORE_SRC_REQUIREMENTS': 'true',
                'SAGEMAKER_DISABLE_BUILT_IN_PROFILER': 'true',
                'SAGEMAKER_DISABLE_DEFAULT_RULES': 'true',
                'SAGEMAKER_TRAINING_JOB_END_DISABLED': 'true',
                'SAGEMAKER_TRAINING_JOB_END_DISABLE': 'true',
                'SAGEMAKER_DEBUG_OUTPUT_DISABLED': 'true',
                'SAGEMAKER_OUTPUT_STRUCTURE_CLEAN': 'true',  # Custom flag for our code
                'SAGEMAKER_PROGRAM': 'scripts/train_multi_model.py',  # Explicitly set the script to run
                # Add new environment variables to enable more optimizations
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',  # Optimize CUDA memory allocation
                'OMP_NUM_THREADS': '4',  # Limit OpenMP thread count
                'MKL_NUM_THREADS': '4',  # Limit MKL thread count
                'NVIDIA_VISIBLE_DEVICES': 'all'  # Ensure all GPUs are visible
            },
            disable_model_download=True,  # Disable model.tar.gz file generation
            disable_output_compression=True  # Disable output.tar.gz file generation
        )
        
        return estimator
    
    # Multi-task subtasks live under ``CSI-Bench/Multitask/<task>/`` in the new
    # S3 layout.  Single-task datasets live directly under ``CSI-Bench/<task>/``.
    _MULTITASK_TASKS = {
        "HumanActivityRecognition",
        "HumanIdentification",
        "ProximityRecognition",
    }

    def _task_s3_path(self, task: str) -> str:
        """Return the S3 prefix that contains the given task's data.

        Supports both the new layout (``s3_data_base/<task>/`` or
        ``s3_data_base/Multitask/<task>/``) and the legacy layout
        (``s3_data_base/tasks/<task>/``).  The new layout is preferred.
        """
        s3_data_base = self.s3_data_base
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        if task in self._MULTITASK_TASKS:
            return f"{s3_data_base}Multitask/{task}/"
        return f"{s3_data_base}{task}/"

    def _prepare_inputs(self, config):
        """Prepare training data input channels for a single task.

        Expected S3 data structure (new ``CSI-Bench`` layout)::

            s3://bucket/path/CSI-Bench/<task>/                  # single-task
            s3://bucket/path/CSI-Bench/Multitask/<task>/        # co-labeled
            s3://bucket/path/CSI-Bench/Multitask/sub_Human_h5/  # shared H5

        When downloaded to a SageMaker instance the data shows up at
        ``/opt/ml/input/data/training/`` and the loader treats that directory
        as ``task_dir`` (via ``--use_root_data_path``).  No ``tasks/`` prefix is
        injected.
        """
        task = config.get('task', config.get('task_name'))
        task_data_path = self._task_s3_path(task)
        print(f"Using task-specific data path: {task_data_path}")
        print(f"Task name: {task} - Data will be downloaded to /opt/ml/input/data/training/")

        input_data = {
            'training': TrainingInput(
                s3_data=task_data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix'
            )
        }

        # For multitask single-task training (e.g. Transformer-only Tab. 4
        # baseline) we also need the shared ``sub_Human_h5/`` data, which lives
        # one level above the per-task metadata.  Mount the whole Multitask/
        # parent so the ``../../sub_Human_h5/...`` paths in the metadata
        # resolve correctly.
        if task in self._MULTITASK_TASKS:
            s3_data_base = self.s3_data_base.rstrip('/') + '/'
            input_data['training'] = TrainingInput(
                s3_data=f"{s3_data_base}Multitask/",
                content_type='application/x-recordio',
                s3_data_type='S3Prefix',
            )

        return input_data
    
    def run_multitask(self, tasks=None, model_type="transformer", override_config=None):
        """
        Run multi-task learning job
        
        Parameters:
            tasks (list): List of tasks to process
            model_type (str): Model type to use
            override_config (dict): Configuration override parameters
            
        Returns:
            dict: Dictionary containing job information
        """
        print(f"Starting multi-task learning job...")
        
        # Create multi-task specific configuration
        multi_config = dict(self.config)
        if override_config:
            multi_config.update(override_config)
        
        # Set model and tasks
        multi_config['model'] = model_type
        
        # Ensure tasks are provided and formatted correctly
        if tasks:
            # If provided as a string, convert to list
            if isinstance(tasks, str):
                tasks = tasks.split(',')
            multi_config['tasks'] = ','.join(tasks)
        elif 'tasks' not in multi_config:
            # If not specified, use first two tasks by default
            available_tasks = multi_config.get('available_tasks', [])
            if len(available_tasks) >= 2:
                multi_config['tasks'] = ','.join(available_tasks[:2])
            else:
                print("Error: Multi-task learning requires at least two tasks")
                return None
        
        # Create estimator for multi-task learning
        estimator = self._create_multitask_estimator(multi_config)
        
        # Get tasks string and replace underscores with dashes for job name
        tasks_str = multi_config.get('tasks', '').replace('_', '-')
        
        # Create job name that complies with SageMaker naming rules
        job_name = f"multitask-{tasks_str}-{self.timestamp}"
        
        # Check if job name exceeds SageMaker's length limit (63 characters)
        if len(job_name) > 63:
            # If too long, use a shorter version (just use timestamp)
            job_name = f"multitask-{self.timestamp}"
        
        # Launch training job
        estimator.fit(
            inputs=self._prepare_multitask_inputs(multi_config),
            job_name=job_name,
            wait=False,
            logs=False  # Don't stream logs automatically
        )
        
        # Return job information
        job_info = {
            'job_name': job_name,
            'tasks': multi_config['tasks'],
            'model': model_type,
            'status': 'InProgress',
            'estimator': estimator
        }
        
        print(f"Multi-task job submitted: {job_name}")
        return job_info
    
    def _create_multitask_estimator(self, config):
        """
        Create SageMaker PyTorch Estimator for multi-task learning
        """
        # Get instance type (may be a single value or from a list per task)
        instance_type = config.get('instance_type', 'ml.g4dn.2xlarge')
        # Ensure instance_type is a string, not a list
        if isinstance(instance_type, list):
            # For multitask, use the most powerful instance in the list
            instance_type = instance_type[0]
            print(f"For multitask learning, using the first instance type from list: {instance_type}")
        
        # Prepare hyperparameters - Use dashes instead of underscores for parameter names.
        # ``pipeline`` is intentionally NOT a hyperparameter: SageMaker would
        # try to pass ``--pipeline multitask`` to ``train_multitask_adapter.py``
        # which uses ``parse_args()`` (strict) and would reject it.  Dispatch
        # to the multitask script via ``SAGEMAKER_PROGRAM`` instead.
        hyperparameters = {
            'tasks': config.get('tasks'),
            'model': config.get('model', 'transformer'),
            # SageMaker mounts the training channel at /opt/ml/input/data/training,
            # which (per ``_prepare_multitask_inputs``) is the contents of
            # ``s3_data_base/Multitask/``.  The multitask script's default
            # ``wifi_benchmark_dataset`` is meaningless inside the container.
            'data-dir': '/opt/ml/input/data/training',
            'save-dir': '/opt/ml/model',
            'win-len': config.get('win_len', 500),
            'feature-size': config.get('feature_size', 232),
            'batch-size': config.get('batch_size', 16),
            'epochs': config.get('epochs', 100),
            'test-splits': config.get('test_splits', 'all'),
            'seed': config.get('seed', 42),
        }
        
        # Add few-shot parameters (if enabled)
        if config.get('enable_few_shot', False) or config.get('fewshot_eval_shots', False):
            hyperparameters['enable-few-shot'] = "True"
            hyperparameters['k-shot'] = config.get('k_shot', 5)
            hyperparameters['inner-lr'] = config.get('inner_lr', 0.01)
            hyperparameters['num-inner-steps'] = config.get('num_inner_steps', 10)
        
        # Add model-specific parameters (if exists)
        if 'model_params' in config:
            for key, value in config['model_params'].items():
                # Ensure parameter names use dashes, not underscores
                fixed_key = key.replace('_', '-')
                hyperparameters[fixed_key] = value
        else:
            # Add common parameters
            for param in ['lr', 'emb_dim', 'dropout', 'patience']:
                if param in config:
                    # Convert parameter name to use dashes
                    fixed_param = param.replace('_', '-')
                    hyperparameters[fixed_param] = config[param]
        
        # Log instance type
        print(f"Creating multitask estimator with instance type: {instance_type}")
        
        # Create estimator
        estimator = PyTorch(
            entry_point='scripts/entry_script.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=config.get('framework_version', '1.12.1'),
            py_version=config.get('py_version', 'py38'),
            instance_count=config.get('instance_count', 1),
            instance_type=instance_type,
            base_job_name='wifi-sensing-multitask',
            hyperparameters=hyperparameters,
            max_run=config.get('max_run', 24 * 3600),  # Default 24-hour maximum run time
            keep_alive_period_in_seconds=config.get('keep_alive_period', 1200),  # Default keep instance active 20 minutes
            disable_model_download=True,  # Disable model.tar.gz file generation
            disable_output_compression=True,  # Disable output.tar.gz file generation
            output_path=self.s3_output_base,  # Explicitly set output path
            code_location=self.s3_output_base,  # Where to store the code package
            debugger_hook_config=False,  # Disable debugger hooks completely
            disable_upload_notifications=True,  # Disable notifications about uploads
            profiler_config=None,  # No profiler config
            environment={
                'SAGEMAKER_S3_OUTPUT': self.s3_output_base,  # Set environment variable for S3 output path
                'SMDEBUG_DISABLED': 'true',
                'SM_DISABLE_DEBUGGER': 'true',
                'SMDATAPARALLEL_DISABLE_DEBUGGER': 'true',
                'SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT': 'true',
                'SMPROFILER_DISABLED': 'true',
                'SM_SMDEBUG_DISABLED': 'true',
                'SM_SMDDP_DISABLE_PROFILING': 'true',
                'SAGEMAKER_DISABLE_PROFILER': 'true',
                'SAGEMAKER_DISABLE_SOURCEDIR': 'true',
                'SAGEMAKER_CONTAINERS_IGNORE_SRC_REQUIREMENTS': 'true',
                'SAGEMAKER_DISABLE_BUILT_IN_PROFILER': 'true',
                'SAGEMAKER_DISABLE_DEFAULT_RULES': 'true',
                'SAGEMAKER_TRAINING_JOB_END_DISABLED': 'true',
                'SAGEMAKER_TRAINING_JOB_END_DISABLE': 'true',
                'SAGEMAKER_DEBUG_OUTPUT_DISABLED': 'true',
                'SAGEMAKER_OUTPUT_STRUCTURE_CLEAN': 'true',  # Custom flag for our code
                'SAGEMAKER_PROGRAM': 'scripts/train_multitask_adapter.py'  # Multi-task LoRA-adapter training entry
            }
        )
        
        return estimator
    
    def _prepare_multitask_inputs(self, config):
        """Prepare input data channels for multi-task training.

        New ``CSI-Bench`` layout: the three multi-task sub-tasks
        (``HumanActivityRecognition``, ``HumanIdentification``,
        ``ProximityRecognition``) plus the shared ``sub_Human_h5/`` and
        ``sub_Human_mat/`` directories all live under
        ``CSI-Bench/Multitask/``.  Mount that parent directory so each
        per-task metadata's ``../../sub_Human_h5/...`` references resolve.
        """
        s3_data_base = self.s3_data_base
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'

        tasks_str = config.get('tasks', '')
        if isinstance(tasks_str, list):
            tasks_list = [t.strip() for t in tasks_str if t and str(t).strip()]
        else:
            tasks_list = [t.strip() for t in tasks_str.split(',') if t.strip()]

        data_path = f"{s3_data_base}Multitask/"
        if tasks_list:
            print(
                f"Multi-task learning using {len(tasks_list)} tasks "
                f"({', '.join(tasks_list)}); data path: {data_path}"
            )
        else:
            print(f"Warning: No tasks specified; defaulting to data path {data_path}")

        input_data = {
            'training': TrainingInput(
                s3_data=data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix',
            )
        }
        return input_data

def run_from_config(config_path=None):
    """
    Run SageMaker training job based on configuration file
    
    Parameters:
        config_path: Path to configuration file, if None use default configuration
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create SageMaker runner
    runner = SageMakerRunner(config)
    
    # Get tasks and pipeline from configuration
    tasks = config.get('task')
    pipeline = config.get('pipeline', 'supervised')
    
    # Run corresponding task based on pipeline type in configuration
    if pipeline == 'multitask':
        # For multi-task learning, ensure tasks parameter is provided
        if 'tasks' in config:
            tasks = config['tasks']
        else:
            # If tasks not specified but task is provided, use task
            if tasks:
                tasks = [tasks]
            else:
                # Use default list of first two tasks
                tasks = config.get('available_tasks', [])[:2]
        
        # Run multi-task learning
        model_type = config.get('model', 'transformer')
        result = runner.run_multitask(tasks=tasks, model_type=model_type)
    else:
        # For supervised learning, can run single task or multiple tasks
        if tasks and not isinstance(tasks, list):
            tasks = [tasks]
        
        # Run batch processing tasks always using all available models from config
        result = runner.run_batch_by_task(tasks=tasks)
    
    return result

def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline in SageMaker')
    
    # Configuration file parameter
    parser.add_argument('--config', type=str, default=None,
                        help='JSON configuration file path')
    
    # Task parameter - this can still be useful to override the config
    parser.add_argument('--task', '--tasks', type=str, default=None,
                        help='Task(s) to run, comma-separated')
    
    args = parser.parse_args()
    
    # If task is specified on command line, override config
    if args.task:
        # Create configuration
        config = load_config(args.config)
        
        # Override task in config
        tasks = [t.strip() for t in args.task.split(',')]
        if len(tasks) == 1:
            config['task'] = tasks[0]
        else:
            config['tasks'] = ','.join(tasks)
        
        # Create SageMaker runner and run batch
        runner = SageMakerRunner(config)
        runner.run_batch_by_task(tasks=tasks)
    else:
        # Run from configuration file
        run_from_config(args.config)

if __name__ == "__main__":
    main()
