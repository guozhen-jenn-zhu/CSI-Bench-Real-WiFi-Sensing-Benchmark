#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script serves as the main entry point for WiFi sensing benchmark.
It incorporates functionality from train.py, run_model.py, and the original local_runner.py.

Configuration File Management:
1. The configs folder now only contains template configuration files
2. Generated configuration files are saved to the results folder using a unified directory structure: results/TASK/MODEL/EXPERIMENT_ID/
   - Supervised learning: results/TASK/MODEL/EXPERIMENT_ID/supervised_config.json
   - Multitask learning: results/TASK/MODEL/EXPERIMENT_ID/multitask_config.json
3. All runtime parameters should be loaded from the configuration file, command-line arguments are no longer used

Usage:
    python local_runner.py --config_file [config_path]
    
Additional parameters:
    --config_file: JSON configuration file to use for all settings
"""

import os
import sys
import subprocess
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import time
import argparse
import json
from datetime import datetime
import importlib.util
import pandas as pd

# Fix encoding issues on Windows
import io
import locale

# Try to set UTF-8 mode for Windows
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

# Set stdout encoding to UTF-8 if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"root_dir is {ROOT_DIR}")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "local_default_config.json")

# Ensure results directory exists
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

def validate_config(config, required_fields=None):
    """
    Validate if the configuration contains all necessary parameters
    
    Args:
        config: Configuration dictionary
        required_fields: List of required fields, if None use default required fields
        
    Returns:
        True if validation succeeds, False otherwise
    """
    if required_fields is None:
        # Define basic required fields
        required_fields = [
            "pipeline", "training_dir", "output_dir", 
            "win_len", "feature_size", "batch_size", "epochs"
        ]
        
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)

    # Special handling for task and tasks parameters.  ``available_tasks`` is
    # also accepted (the runner iterates over each task in turn).
    if (
        "task" not in config
        and "tasks" not in config
        and "available_tasks" not in config
    ):
        missing_fields.append("task or tasks or available_tasks")
    
    if missing_fields:
        print(f"Error: Configuration file is missing the following required parameters: {', '.join(missing_fields)}")
        return False
    
    # Validate if pipeline is valid - hardcoded valid options
    valid_pipelines = ["supervised", "multitask"]
    if config["pipeline"] not in valid_pipelines:
        print(f"Error: Invalid pipeline value: '{config['pipeline']}'")
        print(f"Available options: {valid_pipelines}")
        return False
    
    # Special validation for multitask mode
    if config["pipeline"] == "multitask" and "tasks" not in config:
        print("Error: Multitask pipeline requires 'tasks' parameter")
        return False
        
    # Special validation for supervised mode
    if (
        config["pipeline"] == "supervised"
        and "task" not in config
        and "available_tasks" not in config
    ):
        print("Error: Supervised pipeline requires 'task' or 'available_tasks' parameter")
        return False

    return True

# Load configuration from JSON file
def load_config(config_path=None):
    """Load configuration from JSON file"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate if the configuration file contains all necessary parameters
        if not validate_config(config):
            sys.exit(1)
            
        print(f"Loaded configuration from {config_path}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load config file: {e}")
        sys.exit(1)

# Load the configuration
CONFIG = load_config(DEFAULT_CONFIG_PATH)

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("CUDA not available. Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Neither CUDA nor MPS available. Using CPU.")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Set device string for command line arguments
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

def run_command(cmd, display_output=True, timeout=1800):
    """
    Run command and display output in real-time with timeout handling.
    
    Args:
        cmd: Command to execute
        display_output: Whether to display command output
        timeout: Command execution timeout in seconds, default 30 minutes
        
    Returns:
        Tuple of (return_code, output_string)
    """
    try:
        # Start process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=True
        )
        
        # For storing output
        output = []
        start_time = time.time()
        
        # Main loop
        while process.poll() is None:
            # Check for timeout
            if timeout and time.time() - start_time > timeout:
                if display_output:
                    print(f"\nError: Command execution timed out ({timeout} seconds), terminating...")
                process.kill()
                return -1, '\n'.join(output + [f"Error: Command execution timed out ({timeout} seconds)"])
            
            # Read output line by line without blocking
            try:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    if display_output:
                        print(line)
                    output.append(line)
                else:
                    # Small sleep to reduce CPU usage
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error reading output: {str(e)}")
                time.sleep(0.1)
        
        # Ensure all remaining output is read
        remaining_output, _ = process.communicate()
        if remaining_output:
            for line in remaining_output.splitlines():
                if display_output:
                    print(line)
                output.append(line)
                
        return process.returncode, '\n'.join(output)
        
    except KeyboardInterrupt:
        # User interruption
        if 'process' in locals() and process.poll() is None:
            print("\nUser interrupted, terminating process...")
            process.kill()
        return -2, "User interrupted execution"
        
    except Exception as e:
        # Other exceptions
        error_msg = f"Error executing command: {str(e)}"
        if display_output:
            print(f"\nError: {error_msg}")
        
        # Kill process if still running
        if 'process' in locals() and process.poll() is None:
            process.kill()
        
        return -1, error_msg

def _supervised_run_completed(output_dir, task, model, seed):
    """Return True if a previous successful supervised run exists for
    ``(task, model, seed)`` under ``output_dir``.

    ``run_supervised_direct`` only writes ``supervised_config.json`` after the
    child process exits with code 0, so its presence (with a matching ``seed``)
    is a reliable completion marker.
    """
    model_dir = os.path.join(output_dir, task, model)
    if not os.path.isdir(model_dir):
        return False
    for entry in os.listdir(model_dir):
        cfg_path = os.path.join(model_dir, entry, "supervised_config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, 'r') as f:
                saved_cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if int(saved_cfg.get('seed', -1)) == int(seed):
            return True
    return False


def _multitask_run_completed(output_dir, model, seed):
    """Return True if a previous successful multitask run exists for
    ``(model, seed)`` under ``output_dir``.

    ``run_multitask_direct`` only writes ``multitask_config.json`` after a
    return code of 0, so its presence (with a matching ``seed``) is a reliable
    completion marker.
    """
    model_dir = os.path.join(output_dir, "multitask", model)
    if not os.path.isdir(model_dir):
        return False
    for entry in os.listdir(model_dir):
        cfg_path = os.path.join(model_dir, entry, "multitask_config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, 'r') as f:
                saved_cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if int(saved_cfg.get('seed', -1)) == int(seed):
            return True
    return False


def get_supervised_config(custom_config=None):
    """
    Get configuration for supervised learning pipeline.
    
    Args:
        custom_config: Custom configuration dictionary
        
    Returns:
        Configuration dictionary
    """
    # Custom configuration must be provided
    if custom_config is None:
        print("Error: Configuration parameters must be provided!")
        sys.exit(1)
    
    # Create configuration dictionary
    config = {
        # Data parameters
        'training_dir': custom_config['training_dir'],
        'test_dirs': custom_config.get('test_dirs', []),
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config['model']}_{custom_config['task'].lower()}",
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        
        # Training parameters
        'batch_size': custom_config['batch_size'],
        'learning_rate': custom_config.get('learning_rate', 1e-4),
        'weight_decay': custom_config.get('weight_decay', 1e-5),
        'epochs': custom_config['epochs'],
        'warmup_epochs': custom_config.get('warmup_epochs', 5),
        'patience': custom_config.get('patience', 15),
        
        # Integrated loader options
        'integrated_loader': True,  # Always use integrated loader
        'task': custom_config['task'],
        
        # Other parameters
        'seed': custom_config.get('seed', 42),
        'device': DEVICE,
        'model': custom_config['model'],
        'win_len': custom_config['win_len'],
        'feature_size': custom_config['feature_size'],
        
        # Test split options
        'test_splits': custom_config.get('test_splits', 'all')
    }
    
    # If model_params exists, add it to config
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

def get_multitask_config(custom_config=None):
    """
    Get configuration for multitask learning pipeline
    
    Args:
        custom_config: Custom configuration dictionary
        
    Returns:
        Configuration dictionary
    """
    # Custom configuration must be provided
    if custom_config is None:
        print("Error: Configuration parameters must be provided!")
        sys.exit(1)
    
    # Ensure tasks parameter is available
    if 'tasks' not in custom_config:
        print("Error: 'tasks' parameter is not specified in configuration!")
        sys.exit(1)
        
    # Extract tasks and convert to correct format
    tasks = custom_config.get('tasks')
    if isinstance(tasks, str):
        # If it's a string, it might be a comma-separated list
        custom_config['tasks'] = tasks.split(',')
    elif not isinstance(tasks, list) or not tasks:
        print("Error: 'tasks' parameter should be either a list or a comma-separated string!")
        sys.exit(1)
        
    # Set default task name for directory structure
    custom_config['task'] = 'multitask'
    
    # Create configuration dictionary
    config = {
        # Data parameters
        'training_dir': custom_config['training_dir'],
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config['model']}_multitask",
        
        # Training parameters
        'batch_size': custom_config['batch_size'],
        'learning_rate': custom_config.get('learning_rate', 5e-4),
        'weight_decay': custom_config.get('weight_decay', 1e-5),
        'epochs': custom_config['epochs'],
        'win_len': custom_config['win_len'],
        'feature_size': custom_config['feature_size'],
        
        # Model parameters
        'model': custom_config['model'],
        'emb_dim': custom_config.get('emb_dim', 128),
        'dropout': custom_config.get('dropout', 0.1),
        
        # Task parameters
        'task': custom_config['task'],  # 'multitask' for directory structure
        'tasks': custom_config['tasks'],
    }
    
    # If transformer_config.json exists, try to load it
    transform_path = os.path.join(CONFIG_DIR, "transformer_config.json")
    if os.path.exists(transform_path):
        print(f"Using existing configuration file: {transform_path}")
        with open(transform_path, 'r') as f:
            transformer_config = json.load(f)
            for k, v in transformer_config.items():
                if k in config:
                    config[k] = v
    
    # Ensure tasks parameter is valid and has the correct format
    if not config.get('tasks'):
        print("Error: Multitask configuration must specify 'tasks' parameter!")
        sys.exit(1)
    
    # If model_params exists, add it to config
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

# Thread-safe lock for parent-process stdout when running jobs in parallel.
_PRINT_LOCK = threading.Lock()


def _safe_print(msg):
    """Print ``msg`` atomically across worker threads."""
    with _PRINT_LOCK:
        print(msg, flush=True)


def _tail_log(path, max_chars=400):
    """Return the last non-empty line of ``path`` (or empty string on error)."""
    try:
        with open(path, 'rb') as f:
            try:
                f.seek(-max_chars, os.SEEK_END)
            except OSError:
                f.seek(0)
            tail = f.read().decode('utf-8', errors='replace')
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if line:
                return line
    except (OSError, ValueError):
        pass
    return ''


def _heartbeat_loop(stop_event, interval, status):
    """Periodically print a parent-side heartbeat with job progress.

    Designed for SageMaker / long-running boxes: the parent process never goes
    silent for more than ``interval`` seconds, which (a) makes progress
    visible without waiting for a job to finish and (b) keeps idle-shutdown
    watchdogs from killing the instance.

    ``status`` is a dict managed under ``_PRINT_LOCK``::

        {
            'start_time': float,           # seconds since epoch
            'total': int,                  # full sweep size (pending + pre-skipped)
            'pending': int,                # jobs we actually dispatch this invocation
            'running': dict[str, dict],    # key -> {'gpu_id': int, 'log_file': str, 'started': float}
            'finished': int,               # naturally finished this invocation (success or failure)
            'skipped': int,                # pre-skipped via --skip-existing
        }
    """
    if interval <= 0:
        return
    while not stop_event.wait(interval):
        with _PRINT_LOCK:
            elapsed = time.time() - status['start_time']
            hrs, rem = divmod(int(elapsed), 3600)
            mins = rem // 60
            running = list(status['running'].items())
            finished = status['finished']
            skipped = status['skipped']
            pending = status['pending']
            total = status['total']
            queued = max(0, pending - finished - len(running))
            print(
                f"[heartbeat T+{hrs}h{mins:02d}m] "
                f"running={len(running)} finished={finished} queued={queued} "
                f"(of {pending} pending; skipped={skipped}, total={total})",
                flush=True,
            )
            for key, info in running:
                run_for = int(time.time() - info['started'])
                rh, rrem = divmod(run_for, 3600)
                rm = rrem // 60
                tail = _tail_log(info['log_file']) if info.get('log_file') else ''
                tail_disp = f" :: {tail}" if tail else ''
                print(
                    f"  - [gpu {info['gpu_id']}] {key}  (running {rh}h{rm:02d}m){tail_disp}",
                    flush=True,
                )


def _execute_training_subprocess(cmd, env=None, log_file=None, line_prefix=None):
    """Run ``cmd`` in a subprocess and stream/capture its output.

    Args:
        cmd: shell command string (executed with ``shell=True``).
        env: optional environment dict (e.g. with ``CUDA_VISIBLE_DEVICES`` set).
            If ``None`` the parent process's environment is inherited.
        log_file: if provided, the subprocess output is written to this path
            (one line per readline) and *not* streamed to the parent's stdout.
            If ``None``, output is streamed to stdout exactly like before.
        line_prefix: optional string prepended to each line we print to stdout
            (useful to disambiguate concurrent jobs, e.g. ``"[gpu 2] "``).

    Returns:
        ``(return_code, experiment_id_or_None)`` -- ``experiment_id`` is parsed
        from the first ``"Experiment ID:"`` line in the child's output.
    """
    # Always force unbuffered stdout in the child python.  Without this,
    # ``print()`` in the training script is block-buffered (Python's default
    # when stdout is a pipe), so the per-job log file looks frozen for tens of
    # KB at a time and ``tail -f`` is useless.
    if env is None:
        env = os.environ.copy()
    else:
        env = dict(env)
    env.setdefault('PYTHONUNBUFFERED', '1')

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True,
        env=env,
        bufsize=1,  # line-buffered on the parent side as well
    )

    experiment_id = None
    log_fh = open(log_file, 'w') if log_file else None
    try:
        for line in iter(process.stdout.readline, ""):
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()
            else:
                if line_prefix:
                    # Preserve trailing newline if present
                    sys.stdout.write(line_prefix + line)
                else:
                    sys.stdout.write(line)
                sys.stdout.flush()
            if experiment_id is None and "Experiment ID:" in line:
                experiment_id = line.split("Experiment ID:")[1].strip()
    finally:
        if log_fh is not None:
            log_fh.close()

    return_code = process.wait()
    return return_code, experiment_id


def run_supervised_direct(config, env=None, log_file=None, line_prefix=None):
    """
    Run supervised learning pipeline directly.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code from the process
    """
    # Get necessary parameters
    task_name = config.get('task')
    model_name = config.get('model')
    training_dir = config.get('training_dir')
    base_output_dir = config.get('output_dir')
    
    # Build basic command with proper path quoting for Windows
    executable = f'"{sys.executable}"' if ' ' in sys.executable else sys.executable
    script_path = f'"{os.path.join(SCRIPT_DIR, "train_supervised.py")}"' if ' ' in SCRIPT_DIR else os.path.join(SCRIPT_DIR, 'train_supervised.py')
    
    # Start building command
    cmd = f"{executable} {script_path}"
    
    # Properly quote paths that might contain spaces
    quoted_training_dir = f'"{training_dir}"' if ' ' in training_dir else f'"{training_dir}"'
    quoted_output_dir = f'"{base_output_dir}"' if ' ' in base_output_dir else f'"{base_output_dir}"'
    
    cmd += f" --data_dir={quoted_training_dir}"
    cmd += f" --task_name={task_name}"
    cmd += f" --model={model_name}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"
    cmd += f" --save_dir={quoted_output_dir}"
    cmd += f" --output_dir={quoted_output_dir}"

    # DataLoader workers.  Respect the config (defaults to 0 only when unset,
    # which matches the historical behavior).  With multiple sweep jobs on the
    # same box, ``num_workers=0`` is a severe I/O bottleneck -- each job hits
    # the disk single-threadedly and starves its GPU.
    num_workers = int(config.get('num_workers', 0))
    cmd += f" --num_workers={num_workers}"
    cmd += " --use_root_data_path"  # Flag parameter without value
    
    # Disable pin_memory to resolve MPS warnings
    # MPS device doesn't support pin_memory, so we need to explicitly disable it
    cmd += " --no_pin_memory"
    
    # Add test split parameters (if they exist)
    if 'test_splits' in config:
        test_splits = config['test_splits']
        quoted_test_splits = f'"{test_splits}"' if ' ' in str(test_splits) else f'"{test_splits}"'
        cmd += f" --test_splits={quoted_test_splits}"
    
    # Add other model-specific parameters
    important_params = ['learning_rate', 'weight_decay', 'warmup_epochs', 'patience',
                         'emb_dim', 'dropout', 'd_model', 'seed']
    for param in important_params:
        if param in config:
            cmd += f" --{param}={config[param]}"
    
    # Add parameters from model_params
    if 'model_params' in config:
        for key, value in config['model_params'].items():
            cmd += f" --{key}={value}"
    
    # Run command and capture output
    _safe_print(f"Running supervised learning: {cmd}")
    return_code, experiment_id = _execute_training_subprocess(
        cmd, env=env, log_file=log_file, line_prefix=line_prefix,
    )

    if return_code != 0:
        _safe_print(f"Error running supervised learning: return code {return_code}")
    else:
        _safe_print("Supervised learning completed successfully.")

        # If experiment_id was successfully obtained, save configuration directly to experiment directory
        if experiment_id:
            exp_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
            config_filename = os.path.join(exp_dir, "supervised_config.json")

            try:
                os.makedirs(exp_dir, exist_ok=True)
                with open(config_filename, 'w') as f:
                    json.dump(config, f, indent=2)
                _safe_print(f"Configuration saved to model directory: {config_filename}")
            except Exception as e:
                _safe_print(f"Error saving configuration file: {str(e)}")

    return return_code

def run_multitask_direct(config, env=None, log_file=None, line_prefix=None):
    """
    Run multitask learning pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code (0 for success, non-zero for failure)
    """
    print("Running multitask learning with the following configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Get task parameters
    tasks = config.get('tasks')
    if not tasks:
        print("Error: 'tasks' parameter is missing or empty. Please specify at least one task.")
        return 1
    
    # Ensure tasks has the correct format - should be a comma-separated string without spaces
    if isinstance(tasks, list):
        tasks = ','.join(tasks)
    
    # Get basic parameters
    task_name = 'multitask'  # Always use 'multitask' for directory structure
    model_name = config.get('model')
    base_output_dir = config.get('output_dir')
    
    # Build command with proper path quoting for Windows
    executable = f'"{sys.executable}"' if ' ' in sys.executable else sys.executable
    script_path = f'"{os.path.join(SCRIPT_DIR, "train_multitask_adapter.py")}"' if ' ' in SCRIPT_DIR else os.path.join(SCRIPT_DIR, 'train_multitask_adapter.py')
    
    # Start building command
    cmd = f"{executable} {script_path}"
    
    # Properly quote tasks and paths that might contain spaces
    quoted_tasks = f'"{tasks}"'
    training_dir = config.get('training_dir')
    quoted_training_dir = f'"{training_dir}"' if ' ' in training_dir else f'"{training_dir}"'
    
    cmd += f" --tasks={quoted_tasks}"
    cmd += f" --model={model_name}"
    cmd += f" --data_dir={quoted_training_dir}"
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"

    # Multi-task script writes ``<save_dir>/<task>/<model>/<exp_id>/`` per task,
    # which is exactly the layout the aggregator expects.  Without this the
    # script falls back to ``PROJECT_ROOT/results/multitask`` and the per-task
    # results never land under the user's ``output_dir``.
    quoted_output_dir = f'"{base_output_dir}"'
    cmd += f" --save_dir={quoted_output_dir}"

    # DataLoader workers -- respect the config (see comment in
    # ``run_supervised_direct``).
    num_workers = int(config.get('num_workers', 0))
    cmd += f" --num_workers={num_workers}"
    cmd += " --use_root_data_path"  # Flag parameter without value
    
    # Disable pin_memory to resolve MPS warnings
    # MPS device doesn't support pin_memory, so we need to explicitly disable it
    cmd += " --no_pin_memory"
    
    # Handle optional parameters from model_params
    if 'model_params' in config:
        model_params = config['model_params']
        for key, value in model_params.items():
            cmd += f" --{key}={value}"
    else:
        # If model_params doesn't exist, handle individual parameters.
        # ``train_multitask_adapter.py`` uses ``--lr``; if config supplies
        # ``learning_rate`` we map it to ``--lr``.
        for param in ['lr', 'emb_dim', 'dropout', 'patience', 'data_key', 'seed',
                      'lora_r', 'lora_alpha', 'lora_dropout']:
            if param in config:
                cmd += f" --{param}={config[param]}"
        if 'lr' not in config and 'learning_rate' in config:
            cmd += f" --lr={config['learning_rate']}"

    # Add test_splits (if they exist)
    if 'test_splits' in config:
        test_splits = config['test_splits']
        quoted_test_splits = f'"{test_splits}"' if ' ' in str(test_splits) else f'"{test_splits}"'
        cmd += f" --test_splits={quoted_test_splits}"
    
    # Run command and capture output
    _safe_print(f"Running command: {cmd}")
    return_code, experiment_id = _execute_training_subprocess(
        cmd, env=env, log_file=log_file, line_prefix=line_prefix,
    )

    if return_code != 0:
        _safe_print(f"Error running multitask learning: return code {return_code}")
    else:
        _safe_print("Multitask learning completed successfully.")

        # If experiment_id was successfully obtained, save configuration directly to experiment directory
        if experiment_id:
            exp_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
            config_filename = os.path.join(exp_dir, "multitask_config.json")

            try:
                os.makedirs(exp_dir, exist_ok=True)
                with open(config_filename, 'w') as f:
                    json.dump(config, f, indent=2)
                _safe_print(f"Configuration saved to model directory: {config_filename}")
            except Exception as e:
                _safe_print(f"Error saving configuration file: {str(e)}")

    return return_code

def main():
    """Main entry point function"""
    # Parse command line arguments - only accept config_file
    parser = argparse.ArgumentParser(description='Run WiFi Sensing Pipeline')
    
    # config_file is the only required parameter (``--config`` is an alias)
    parser.add_argument('--config_file', '--config', dest='config_file',
                        type=str, default=DEFAULT_CONFIG_PATH,
                        help='JSON configuration file for all settings')
    parser.add_argument('--skip-existing', '--skip_existing',
                        dest='skip_existing', action='store_true',
                        help='Skip (task, model, seed) combinations that already '
                             'have a completed run on disk (detected via the '
                             '*_config.json marker written on success).')
    parser.add_argument('--num-gpus', '--num_gpus',
                        dest='num_gpus', type=str, default='1',
                        help='How many GPUs to dispatch sweep jobs across. '
                             "Default is ``1`` (sequential, one job at a time "
                             "on cuda:0) -- same behavior as the original "
                             "runner.  Pass ``auto`` to detect via "
                             '``torch.cuda.device_count()`` or an integer to '
                             'opt into parallel multi-GPU sweeps; each '
                             'concurrent job is then pinned to its own GPU '
                             'via ``CUDA_VISIBLE_DEVICES``.')
    parser.add_argument('--jobs-per-gpu', '--jobs_per_gpu',
                        dest='jobs_per_gpu', type=int, default=1,
                        help='How many concurrent jobs to run *per* GPU '
                             '(default: 1). Increase only if a single run does '
                             'not saturate one GPU.')
    parser.add_argument('--heartbeat-interval', '--heartbeat_interval',
                        dest='heartbeat_interval', type=int, default=300,
                        help='Seconds between parent-side heartbeat lines that '
                             'show running/done/queued and tail the per-job '
                             'logs. Set to 0 to disable. Default: 300 (5 min). '
                             'Helps prevent SageMaker idle-shutdown.')

    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.config_file)

    # CLI flag wins, but allow the config to opt in as well.
    skip_existing = bool(args.skip_existing or config.get('skip_existing', False))

    # Resolve the requested GPU count.  ``auto`` -> torch.cuda.device_count().
    num_gpus_arg = str(args.num_gpus).strip().lower()
    if num_gpus_arg == 'auto':
        requested_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        try:
            requested_gpus = int(num_gpus_arg)
        except ValueError:
            print(f"Error: --num-gpus must be 'auto' or an integer, got {args.num_gpus!r}")
            return 1
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if requested_gpus > available_gpus:
        print(f"Warning: requested {requested_gpus} GPUs but only {available_gpus} "
              "are visible; clamping.")
        requested_gpus = available_gpus
    requested_gpus = max(0, requested_gpus)
    jobs_per_gpu = max(1, int(args.jobs_per_gpu))
    # ``max_workers`` controls parent-side parallelism.  0 GPUs (CPU box) still
    # runs sequentially.
    max_workers = max(1, requested_gpus * jobs_per_gpu) if requested_gpus > 0 else 1
    
    # Extract pipeline type from configuration
    pipeline = config.get('pipeline')
    
    # Ensure pipeline value is valid
    valid_pipelines = ["supervised", "multitask"]
    if pipeline not in valid_pipelines:
        print(f"Error: Invalid pipeline value: '{pipeline}'")
        print(f"Available options: {valid_pipelines}")
        return 1
    
    # Expand user paths (e.g. ``~/Data/CSI-Bench/``) before passing to children
    if 'training_dir' in config:
        config['training_dir'] = os.path.expanduser(config['training_dir'])
        os.environ['WIFI_DATA_DIR'] = config['training_dir']
    if 'output_dir' in config:
        config['output_dir'] = os.path.expanduser(config['output_dir'])

    # Get all available models
    available_models = config.get('available_models', [])
    if not available_models:
        print("Warning: No available models specified in configuration. Using default model 'mlp'.")
        available_models = ['mlp']

    # Get list of seeds (single-seed sweep by default; multiple seeds opt-in).
    seeds = config.get('seeds')
    if seeds is None:
        seeds = [config.get('seed', 42)]
    if isinstance(seeds, str):
        seeds = [int(s.strip()) for s in seeds.split(',') if s.strip()]
    elif isinstance(seeds, int):
        seeds = [seeds]
    seeds = [int(s) for s in seeds]

    # Get list of tasks for supervised mode (multitask consumes the ``tasks``
    # key as one job and is not looped per-task here).
    if pipeline == 'multitask':
        task_list = [None]  # placeholder; multitask reads ``tasks`` directly
    else:
        task_list = config.get('available_tasks') or [config.get('task')]
        task_list = [t for t in task_list if t]

    # Record results across (task, model, seed)
    results = {}

    # First pass: build the full list of pending jobs (applying --skip-existing
    # up front so the dispatcher only sees real work).
    jobs = []  # list of dicts: {key, task, model, seed, pipeline_config}
    for task in task_list:
        for model in available_models:
            for seed in seeds:
                if pipeline == 'multitask':
                    key = f"multitask/{model}/seed{seed}"
                else:
                    key = f"{task}/{model}/seed{seed}"

                # Optional resume: skip combinations that already have a
                # completed run on disk (detected via the success-only
                # ``*_config.json`` marker written by ``run_*_direct``).
                if skip_existing:
                    if pipeline == 'multitask':
                        already_done = _multitask_run_completed(
                            config['output_dir'], model, seed)
                    else:
                        already_done = _supervised_run_completed(
                            config['output_dir'], task, model, seed)
                    if already_done:
                        print(f"[skip-existing] {key} already completed; skipping.")
                        results[key] = {
                            'status': 'SKIPPED',
                            'return_code': 0,
                            'run_time': 0.0,
                        }
                        continue

                # Create a new config copy for each (task, model, seed)
                run_config = config.copy()
                run_config['model'] = model
                run_config['seed'] = seed
                if pipeline != 'multitask' and task is not None:
                    run_config['task'] = task

                # Get specific pipeline configuration
                if pipeline == 'multitask':
                    pipeline_config = get_multitask_config(run_config)
                else:
                    pipeline_config = get_supervised_config(run_config)

                jobs.append({
                    'key': key,
                    'task': task,
                    'model': model,
                    'seed': seed,
                    'pipeline_config': pipeline_config,
                })

    print(f"\n{'='*60}")
    print(f"Dispatching {len(jobs)} job(s) across {requested_gpus} GPU(s) "
          f"(jobs_per_gpu={jobs_per_gpu}, max_workers={max_workers}, "
          f"heartbeat={args.heartbeat_interval}s)")
    print(f"{'='*60}\n")

    # Shared status used by the heartbeat thread.  Mutations happen under
    # ``_PRINT_LOCK`` so the heartbeat sees a consistent snapshot.
    pre_skipped = sum(1 for r in results.values() if r.get('status') == 'SKIPPED')
    status = {
        'start_time': time.time(),
        'total': len(jobs) + pre_skipped,   # full sweep size
        'pending': len(jobs),               # actual jobs to run this invocation
        'running': {},                      # key -> {'gpu_id', 'log_file', 'started'}
        'finished': 0,                      # naturally finished this invocation
        'skipped': pre_skipped,
    }

    def _run_one(job, gpu_id=None):
        """Run a single job, optionally pinned to ``gpu_id`` (CUDA_VISIBLE_DEVICES)."""
        key = job['key']
        pipeline_config = job['pipeline_config']

        if gpu_id is not None:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            line_prefix = f"[gpu {gpu_id} {key}] "
            # Stream per-job output to a log file so concurrent jobs don't
            # interleave on stdout.  Logs live alongside the per-run results.
            log_dir = os.path.join(config['output_dir'], '_sweep_logs')
            os.makedirs(log_dir, exist_ok=True)
            safe_key = key.replace('/', '__')
            log_file = os.path.join(
                log_dir,
                f"{safe_key}_gpu{gpu_id}_{int(time.time())}.log",
            )
            _safe_print(f"[gpu {gpu_id}] starting {key} -> {log_file}")
        else:
            env = None
            line_prefix = None
            log_file = None
            _safe_print(f"\n{'='*60}\nStarting training: {key}\n{'='*60}\n")

        start_time = time.time()
        with _PRINT_LOCK:
            status['running'][key] = {
                'gpu_id': gpu_id if gpu_id is not None else -1,
                'log_file': log_file,
                'started': start_time,
            }
        try:
            if pipeline == 'multitask':
                return_code = run_multitask_direct(
                    pipeline_config, env=env, log_file=log_file, line_prefix=line_prefix,
                )
            else:
                return_code = run_supervised_direct(
                    pipeline_config, env=env, log_file=log_file, line_prefix=line_prefix,
                )
        finally:
            with _PRINT_LOCK:
                status['running'].pop(key, None)
                status['finished'] += 1
        end_time = time.time()

        result = {
            'status': 'SUCCESS' if return_code == 0 else 'FAILED',
            'return_code': return_code,
            'run_time': end_time - start_time,
        }
        if gpu_id is not None:
            _safe_print(
                f"[gpu {gpu_id}] finished {key}: "
                f"{'OK' if return_code == 0 else 'FAILED'} "
                f"({(end_time - start_time)/60:.2f} min)"
            )
        else:
            _safe_print(
                f"\n{key}: {'OK' if return_code == 0 else 'FAILED'} "
                f"({(end_time - start_time)/60:.2f} min)"
            )
        return key, result

    # Start a parent-side heartbeat thread when we're running in parallel
    # mode (sequential mode already streams full child output to stdout, so a
    # heartbeat would just be noise).  In parallel mode the parent is otherwise
    # silent for hours, so the heartbeat both surfaces progress AND keeps
    # SageMaker (and other idle-shutdown watchdogs) from killing the instance.
    hb_stop = threading.Event()
    hb_thread = None
    if (max_workers > 1 and args.heartbeat_interval
            and args.heartbeat_interval > 0 and jobs):
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(hb_stop, args.heartbeat_interval, status),
            daemon=True,
        )
        hb_thread.start()

    try:
        # Dispatch the jobs.  Sequential code path when max_workers == 1 so single-
        # GPU users (and CPU users) see unchanged streaming output.
        if max_workers <= 1 or not jobs:
            for job in jobs:
                key, result = _run_one(job, gpu_id=None)
                results[key] = result
        else:
            # GPU slot pool: each worker grabs a free GPU id, runs its job,
            # returns the id.  This guarantees no two concurrent jobs land on
            # the same GPU (subject to ``jobs_per_gpu``).
            gpu_slots = queue.Queue()
            for g in range(requested_gpus):
                for _ in range(jobs_per_gpu):
                    gpu_slots.put(g)

            def _worker(job):
                gpu_id = gpu_slots.get()
                try:
                    return _run_one(job, gpu_id=gpu_id)
                finally:
                    gpu_slots.put(gpu_id)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_worker, j) for j in jobs]
                for fut in as_completed(futures):
                    try:
                        key, result = fut.result()
                    except Exception as exc:
                        _safe_print(f"[scheduler] worker raised: {exc}")
                        continue
                    results[key] = result
    finally:
        hb_stop.set()
        if hb_thread is not None:
            hb_thread.join(timeout=2.0)

    # Print summary of all runs
    print(f"\n{'='*60}")
    print(f"All training completed")
    print(f"{'='*60}")
    print(f"Results summary:")

    successful = 0
    failed = 0
    skipped = 0
    for key, result in results.items():
        status = result['status']
        run_time = result['run_time']
        print(f"  - {key}: {status}, time: {run_time/60:.2f} min")
        if status == 'SUCCESS':
            successful += 1
        elif status == 'SKIPPED':
            skipped += 1
        else:
            failed += 1

    total = len(results)
    print(f"\nSuccessful: {successful}/{total}, Skipped: {skipped}/{total}, "
          f"Failed: {failed}/{total}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
