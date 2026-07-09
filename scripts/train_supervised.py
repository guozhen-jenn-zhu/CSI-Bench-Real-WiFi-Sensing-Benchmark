#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # Ensure pandas is imported
from load.supervised.benchmark_loader import load_benchmark_supervised
from tqdm import tqdm
import json
import hashlib
import time
import matplotlib.pyplot as plt
import warnings

# Mute sklearn warnings about precision/recall being ill-defined
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Import model classes from models.py
from model.supervised.models import (
    MLPClassifier,
    LSTMClassifier,
    ResNet18Classifier,
    TransformerClassifier,
    ViTClassifier,
    PatchTST,
    TimesFormer1D
)
from model.supervised.pruned_attention_gru import PrunedAttentionGRUClassifier
from model.supervised.newmodel import NewModel

# Import TaskTrainer
from engine.supervised.task_trainer import TaskTrainer

# Add few-shot learning import
from engine.few_shot import FewShotAdapter

# Model factory dictionary
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier,
    'patchtst': PatchTST,
    'timesformer1d': TimesFormer1D,
    'pruned_attention_gru': PrunedAttentionGRUClassifier,
    'newmodel': NewModel,
}

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train a supervised model on WiFi benchmark dataset')
        parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                            help='Root directory of the dataset')
        parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                            help='Name of the task to train on')
        parser.add_argument('--model', type=str, default='vit', 
                            choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit', 'patchtst', 'timesformer1d', 'pruned_attention_gru', 'newmodel'],
                            help='Type of model to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key for CSI data in h5 files')
        parser.add_argument('--save_dir', type=str, default='results',
                            help='Directory to save checkpoints')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (defaults to save_dir if not specified)')
        parser.add_argument('--weight_decay', type=float, default=1e-5, 
                            help='Weight decay for optimizer')
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help='Number of warmup epochs')
        parser.add_argument('--patience', type=int, default=15,
                            help='Patience for early stopping')
        # Additional model parameters
        parser.add_argument('--win_len', type=int, default=500, 
                            help='Window length for WiFi CSI data')
        parser.add_argument('--feature_size', type=int, default=232,
                            help='Feature size for WiFi CSI data')
        parser.add_argument('--in_channels', type=int, default=1, 
                            help='Number of input channels for convolutional models')
        parser.add_argument('--emb_dim', type=int, default=128, 
                            help='Embedding dimension for transformer models')
        parser.add_argument('--d_model', type=int, default=256, 
                            help='Model dimension for Transformer model')
        parser.add_argument('--dropout', type=float, default=0.1, 
                            help='Dropout rate for regularization')
        # PatchTST specific parameters
        parser.add_argument('--patch_len', type=int, default=16,
                            help='Patch length for PatchTST model')
        parser.add_argument('--stride', type=int, default=8,
                            help='Stride for patches in PatchTST model')
        parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'],
                            help='Pooling method for PatchTST (cls or mean)')
        parser.add_argument('--head_dropout', type=float, default=0.2,
                            help='Dropout rate for classification head')
        # TimesFormer-1D specific parameters
        parser.add_argument('--patch_size', type=int, default=4,
                            help='Patch size for TimesFormer-1D model')
        parser.add_argument('--attn_dropout', type=float, default=0.1,
                            help='Dropout rate for attention layers')
        parser.add_argument('--mlp_ratio', type=float, default=4.0,
                            help='MLP ratio for transformer blocks')
        parser.add_argument('--depth', type=int, default=6,
                            help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads')
        # PrunedAttentionGRU / NewModel parameters
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='Hidden dim for PAG / NewModel')
        parser.add_argument('--attention_dim', type=int, default=32,
                            help='Attention dim for PrunedAttentionGRU')
        # Testing specific parameters
        parser.add_argument('--test_splits', type=str, default='all',
                            help='Comma-separated list of test splits to use. If "all", all available test splits will be used. '
                                 'Examples: test_id,test_cross_env,test_cross_user,test_cross_device,hard_cases')
        
        # Few-shot learning parameters
        parser.add_argument('--enable_few_shot', action='store_true',
                            help='Enable few-shot learning adaptation for cross-domain scenarios')
        parser.add_argument('--k_shot', type=int, default=5,
                            help='Number of examples per class for few-shot adaptation')
        parser.add_argument('--inner_lr', type=float, default=0.01,
                            help='Learning rate for few-shot adaptation')
        parser.add_argument('--num_inner_steps', type=int, default=10,
                            help='Number of gradient steps for few-shot adaptation')
        
        # Add data loading and optimization options
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of worker processes for data loading')
        parser.add_argument('--use_root_data_path', action='store_true',
                            help='Use root directory as task directory')
        parser.add_argument('--file_format', type=str, default='h5',
                            help='File format for data files (h5, mat, or npy)')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode')
        parser.add_argument('--pin_memory', action='store_true', default=True,
                            help='Enable pin memory for data loading (not recommended for MPS)')
        parser.add_argument('--no_pin_memory', action='store_true',
                            help='Disable pin memory for data loading (use for MPS devices)')
        
        args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.save_dir
    
    # Ensure directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check if running in SageMaker
    is_sagemaker = os.path.exists('/opt/ml/model')
    
    # Generate a unique experiment ID based on parameters hash.  The hash MUST
    # include ``seed`` -- otherwise multi-seed sweeps collide into the same
    # experiment directory and later seeds silently overwrite earlier ones.
    param_str = f"{args.learning_rate}_{args.batch_size}_{args.epochs}_{args.weight_decay}_{args.warmup_epochs}_{args.win_len}_{args.feature_size}_seed{args.seed}"
    if hasattr(args, 'dropout') and args.dropout is not None:
        param_str += f"_{args.dropout}"
    if hasattr(args, 'emb_dim') and args.emb_dim is not None:
        param_str += f"_{args.emb_dim}"
    if hasattr(args, 'd_model') and args.d_model is not None:
        param_str += f"_{args.d_model}"
    if hasattr(args, 'in_channels') and args.in_channels is not None:
        param_str += f"_{args.in_channels}"

    experiment_id = f"params_{hashlib.md5(param_str.encode()).hexdigest()[:10]}"
    
    if is_sagemaker:
        print("Running in SageMaker environment")
        model_dir = '/opt/ml/model'
        # If running in SageMaker, ensure we save in the model directory with task/model/experiment structure
        # Directory structure: model_dir/task_name/model/experiment_id/
        results_dir = os.path.join(model_dir, args.task_name, args.model, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        # Also create the checkpoints directory under the task/model/experiment structure
        # Directory structure: save_dir/task_name/model/experiment_id/
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        print("Running in local environment")
        # Use specified directories for local environment with task/model/experiment structure
        # Directory structure: output_dir/task_name/model/experiment_id/
        results_dir = os.path.join(args.output_dir, args.task_name, args.model, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        # Create the checkpoints directory under the task/model/experiment structure
        # Directory structure: save_dir/task_name/model/experiment_id/
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create best_performance.json file at model level if it doesn't exist
        model_level_dir = os.path.join(args.output_dir, args.task_name, args.model)
        best_performance_path = os.path.join(model_level_dir, "best_performance.json")
        if not os.path.exists(best_performance_path):
            with open(best_performance_path, "w") as f:
                json.dump({
                    "best_test_accuracy": 0.0,
                    "best_test_f1_score": 0.0,
                    "best_experiment_id": None,
                    "best_experiment_params": {}
                }, f, indent=4)
    
    print(f"Experiment ID: {experiment_id}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Model checkpoints will be saved to: {checkpoint_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for available test splits in the dataset.  Search the new
    # ``CSI-Bench/<task>/`` and ``CSI-Bench/Multitask/<task>/`` layouts plus the
    # legacy ``tasks/<task>/`` directory.
    candidate_task_dirs = [
        os.path.join(args.data_dir, args.task_name),
        os.path.join(args.data_dir, "Multitask", args.task_name),
        os.path.join(args.data_dir, "tasks", args.task_name),
        args.data_dir,
    ]
    task_dir = next(
        (p for p in candidate_task_dirs if os.path.isdir(os.path.join(p, "splits"))),
        candidate_task_dirs[0],
    )
    splits_dir = os.path.join(task_dir, "splits")
    available_test_splits = []
    print((f"********************splits_dir is {splits_dir}"))
    # Filter out split files belonging to other (few-shot subset) tasks, e.g.
    # ``train_id_p5.json``, ``test_id_p3.json``, ``p5_info.json``.
    import re as _re
    _ignore_pats = (_re.compile(r"_p\d+$"), _re.compile(r"^p\d+(_|$)"))
    if os.path.exists(splits_dir):
        for filename in os.listdir(splits_dir):
            if filename.endswith(".json"):
                split_name = filename.replace(".json", "")
                if any(p.search(split_name) for p in _ignore_pats):
                    print(f"  Ignoring split file (other task): {filename}")
                    continue
                if split_name.startswith("test_") or split_name == "hard_cases":
                    available_test_splits.append(split_name)
    
    print(f"Available test splits: {available_test_splits}")
    
    # Determine which test splits to use
    if args.test_splits == "all":
        test_splits = available_test_splits
    elif args.test_splits is not None:
        test_splits = args.test_splits.split(',')
        # Validate that the requested splits exist
        for split in test_splits:
            if split not in available_test_splits and split != "test_id":
                print(f"Warning: Requested test split '{split}' not found. Available splits: {available_test_splits}")
    else:
        # Default to test_id (in-distribution test)
        test_splits = ["test_id"]
    
    print(f"Using test splits: {test_splits}")
    
    # Add handling for None values to prevent dataloader errors
    def custom_collate_fn(batch):
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        
        # If no samples remain after filtering, return empty tensors
        if len(batch) == 0:
            return torch.zeros(0, 1, args.win_len, args.feature_size), torch.zeros(0, dtype=torch.long)
        
        # Use default collate function for the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)
    
    # Process pin_memory parameter
    if args.no_pin_memory:
        args.pin_memory = False
    
    # Load data
    print(f"Loading data from {args.data_dir} for task {args.task_name}...")
    data = load_benchmark_supervised(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        file_format=args.file_format,
        data_key=args.data_key,
        num_workers=args.num_workers,
        test_splits=test_splits,
        use_root_as_task_dir=args.use_root_data_path,
        collate_fn=custom_collate_fn,
        pin_memory=args.pin_memory,
        debug=False
    )
    
    # Extract data from the returned dictionary
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # Get training and validation loaders
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    if val_loader is None:
        print("Warning: No validation data found. Using training data for validation.")
        val_loader = train_loader
    
    # Count unique labels in the dataset
    all_labels = []
    dataset = train_loader.dataset
    print(f"Detected {num_classes} classes in the dataset")
    
    # Get test loaders
    test_loaders = {k: v for k, v in loaders.items() if k.startswith('test')}
    if not test_loaders:
        print("Warning: No test splits found in the dataset. Check split names and dataset structure.")
    else:
        print(f"Loaded {len(test_loaders)} test splits: {list(test_loaders.keys())}")
    
    # Prepare model
    print(f"Creating {args.model.upper()} model...")
    ModelClass = MODEL_TYPES[args.model]
    
    model_kwargs = {'num_classes': num_classes}
    
    # Add model-specific parameters
    if args.model in ['mlp', 'vit', 'patchtst', 'timesformer1d']:
        model_kwargs.update({'win_len': args.win_len, 'feature_size': args.feature_size})
    
    # ResNet18 specific parameters
    if args.model == 'resnet18':
        model_kwargs.update({'in_channels': args.in_channels})
    
    # LSTM specific parameters
    if args.model == 'lstm':
        model_kwargs.update({'feature_size': args.feature_size})
    
    # Transformer specific parameters
    if args.model == 'transformer':
        model_kwargs.update({
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'dropout': args.dropout
        })
    
    # ViT specific parameters
    if args.model == 'vit':
        model_kwargs.update({
            'emb_dim': args.emb_dim,
            'dropout': args.dropout
        })
    
    # PatchTST specific parameters
    if args.model == 'patchtst':
        model_kwargs.update({
            'patch_len': args.patch_len,
            'stride': args.stride,
            'emb_dim': args.emb_dim, 
            'pool': args.pool,
            'head_dropout': args.head_dropout,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'dropout': args.dropout
        })
    
    # TimesFormer-1D specific parameters
    if args.model == 'timesformer1d':
        model_kwargs.update({
            'patch_size': args.patch_size,
            'emb_dim': args.emb_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'attn_dropout': args.attn_dropout,
            'head_dropout': args.head_dropout,
            'mlp_ratio': args.mlp_ratio,
            'dropout': args.dropout
        })

    # PrunedAttentionGRU specific parameters
    if args.model == 'pruned_attention_gru':
        model_kwargs.update({
            'feature_size': args.feature_size,
            'win_len': args.win_len,
            'hidden_dim': getattr(args, 'hidden_dim', 128),
            'attention_dim': getattr(args, 'attention_dim', 32),
        })

    # NewModel specific parameters
    if args.model == 'newmodel':
        model_kwargs.update({
            'feature_size': args.feature_size,
            'win_len': args.win_len,
            'hidden_dim': getattr(args, 'hidden_dim', 128),
            'dropout': args.dropout,
        })

    # Initialize model
    model = ModelClass(**model_kwargs)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Create config dictionary for saving
    config = {
        'model': args.model,
        'task': args.task_name,
        'task_name': args.task_name,
        'seed': args.seed,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'test_splits': test_splits
    }
    
    # Save configuration
    config_path = os.path.join(results_dir, f"{args.model}_{args.task_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
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
        save_path=checkpoint_dir,
        num_classes=num_classes,
        config={
            'model': args.model,
            'task': args.task_name,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'patience': args.patience,
        },
        label_mapper=label_mapper
    )
    
    # Train the model with early stopping
    model, training_results = trainer.train()
    best_epoch = training_results['best_epoch']
    history = training_results['training_dataframe']
    
    # Evaluate on test dataset
    print("\nEvaluating on test splits:")
    all_results = {}
    for key, loader in test_loaders.items():
        # Skip splits with no samples (e.g. an empty difficulty tier).  We
        # still record an entry so downstream aggregation knows the split
        # existed and was empty for this (task, model, seed).
        n_samples = len(loader.dataset) if loader is not None else 0
        if n_samples == 0:
            print(f"Skipping {key} split: 0 samples")
            all_results[key] = {
                'loss': None, 'accuracy': None, 'f1_score': None, 'n_samples': 0,
            }
            continue

        print(f"Evaluating on {key} split ({n_samples} samples):")
        loss, accuracy = trainer.evaluate(loader)
        # Use calculate_metrics to get f1_score if available
        try:
            f1_score, _ = trainer.calculate_metrics(loader)
        except Exception as _e:
            print(f"  [warn] calculate_metrics failed on {key}: {_e}")
            f1_score = 0.0

        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'n_samples': n_samples,
        }
        all_results[key] = metrics
        print(f"{key} accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1_score']:.4f}")

        # Generate confusion matrix
        print(f"Generating confusion matrix for {key} split...")
        try:
            trainer.plot_confusion_matrix(data_loader=loader, mode=key)
        except Exception as _e:
            print(f"  [warn] plot_confusion_matrix failed on {key}: {_e}")
    print(f"***********************FEW_SHOT{args.enable_few_shot}")
    # Apply few-shot learning if enabled
    if args.enable_few_shot:
        print("\n=== Few-Shot Learning Adaptation ===")
        
        # Create few-shot adapter
        few_shot_adapter = FewShotAdapter(
            model=model,
            device=device,
            inner_lr=args.inner_lr,
            num_inner_steps=args.num_inner_steps,
            k_shot=args.k_shot
        )
        
        # Apply few-shot adaptation on each cross-domain test split
        fs_results = {}
        for key, loader in test_loaders.items():
            # Skip in-distribution test set (no need for adaptation)
            if key == 'test_id':
                continue
                
            print(f"\nApplying few-shot adaptation on {key}...")
            adaptation_results = few_shot_adapter.adapt_and_evaluate(loader)
            fs_results[key] = adaptation_results
            
            # Print improvement
            if 'improvement' in adaptation_results:
                acc_imp = adaptation_results['improvement']['accuracy']
                f1_imp = adaptation_results['improvement']['f1_score']
                print(f"Few-shot improved {key}: Accuracy +{acc_imp:.4f}, F1-Score +{f1_imp:.4f}")
        
        # Create comparison table for original vs. few-shot adapted
        print("\nComparison: Original vs. Few-Shot Adapted")
        print(f"{'Split':<20} {'Original Acc':<15} {'Adapted Acc':<15} {'Improvement':<15}")
        print('-' * 70)
        
        for split, results in fs_results.items():
            orig_acc = results['original']['accuracy']
            adapted_acc = results['adapted']['accuracy']
            improvement = results['improvement']['accuracy']
            print(f"{split:<20} {orig_acc:.4f}{'':>9} {adapted_acc:.4f}{'':>9} {improvement:+.4f}{'':>9}")
        
        # Plot comparison bar chart
        plt.figure(figsize=(12, 6))
        splits = list(fs_results.keys())
        x = range(len(splits))
        original_accs = [fs_results[split]['original']['accuracy'] for split in splits]
        adapted_accs = [fs_results[split]['adapted']['accuracy'] for split in splits]
        
        width = 0.35
        plt.bar([i - width/2 for i in x], original_accs, width, label='Original')
        plt.bar([i + width/2 for i in x], adapted_accs, width, label='Few-Shot Adapted')
        
        plt.xlabel('Test Split')
        plt.ylabel('Accuracy')
        plt.title(f'Few-Shot Adaptation Results ({args.k_shot}-shot, {args.num_inner_steps} steps)')
        plt.xticks(x, splits)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(original_accs):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
        for i, v in enumerate(adapted_accs):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        few_shot_plot_path = os.path.join(results_dir, f"{args.model}_{args.task_name}_few_shot_comparison.png")
        plt.savefig(few_shot_plot_path)
        print(f"Few-shot comparison plot saved to: {few_shot_plot_path}")
        
        # Save few-shot results to JSON
        fs_results_serializable = {}
        for split, res in fs_results.items():
            # Prepare a dict with all necessary values and ensure they're JSON serializable
            original_dict = {}
            for k, v in res['original'].items():
                if k != 'confusion_matrix':
                    if isinstance(v, np.ndarray):
                        original_dict[k] = v.tolist()
                    elif isinstance(v, (np.int32, np.int64)):
                        original_dict[k] = int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        original_dict[k] = float(v)
                    else:
                        original_dict[k] = v
            
            adapted_dict = {}
            for k, v in res['adapted'].items():
                if k != 'confusion_matrix':
                    if isinstance(v, np.ndarray):
                        adapted_dict[k] = v.tolist()
                    elif isinstance(v, (np.int32, np.int64)):
                        adapted_dict[k] = int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        adapted_dict[k] = float(v)
                    else:
                        adapted_dict[k] = v
            
            improvement_dict = {}
            for k, v in res['improvement'].items():
                if isinstance(v, np.ndarray):
                    improvement_dict[k] = v.tolist()
                elif isinstance(v, (np.int32, np.int64)):
                    improvement_dict[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    improvement_dict[k] = float(v)
                else:
                    improvement_dict[k] = v
            
            # Now create the final dict with all JSON serializable values
            fs_results_serializable[split] = {
                'original': original_dict,
                'adapted': adapted_dict,
                'improvement': improvement_dict,
                'support_set_size': res.get('support_set_size', args.k_shot * num_classes),
                'k_shot': args.k_shot,
                'inner_lr': args.inner_lr,
                'num_inner_steps': args.num_inner_steps
            }
        
        few_shot_results_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_few_shot_results.json")
        with open(few_shot_results_file, 'w') as f:
            json.dump(fs_results_serializable, f, indent=4)
        print(f"Few-shot results saved to: {few_shot_results_file}")
    
    # Create a summary table with all test results
    summary_table = []
    for split_name, metrics in all_results.items():
        # ``metrics`` may carry None values for empty splits.
        acc = metrics.get('accuracy')
        f1 = metrics.get('f1_score')
        row = {
            'Split': split_name,
            'Accuracy': f"{acc:.4f}" if isinstance(acc, (int, float)) else 'N/A',
            'F1-Score': f"{f1:.4f}" if isinstance(f1, (int, float)) else 'N/A',
        }
        summary_table.append(row)
    
    # Add few-shot results to summary if applicable
    if args.enable_few_shot:
        for split_name, results in fs_results.items():
            row = {
                'Split': f"{split_name} (few-shot)",
                'Accuracy': f"{results['adapted']['accuracy']:.4f}",
                'F1-Score': f"{results['adapted']['f1_score']:.4f}"
            }
            summary_table.append(row)
    
    # Print summary table
    print("\nTest Results Summary:")
    print(f"{'Split':<25} {'Accuracy':<15} {'F1-Score':<15}")
    print('-' * 55)
    for row in summary_table:
        print(f"{row['Split']:<25} {row['Accuracy']:<15} {row['F1-Score']:<15}")
    
    # Save test results
    results_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_results.json")
    
    # Some objects in the metrics might not be JSON serializable, so we need to convert them
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Process all keys and values in all_results
    for key in all_results:
        all_results[key] = {k: convert_to_json_serializable(v) for k, v in all_results[key].items()}
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Save summary that includes training history, best epoch
    summary = {
        'best_epoch': best_epoch,
        'best_val_loss': float(history.iloc[best_epoch-1]['Val Loss']),
        'best_val_accuracy': float(history.iloc[best_epoch-1]['Val Accuracy']),
        'experiment_id': experiment_id,
        'experiment_completed': True
    }
    
    # Add test results to summary
    for split_name, metrics in all_results.items():
        summary[f'{split_name}_accuracy'] = metrics['accuracy']
        summary[f'{split_name}_f1_score'] = metrics['f1_score']
    
    summary_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save training history
    history_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_train_history.csv")
    history.to_csv(history_file, index=False)
    
    print(f"\nTraining and evaluation completed.")
    print(f"Best model from epoch {best_epoch}, saved to {checkpoint_dir}")
    print(f"Results saved to {results_dir}")
    
    # Check if this is the best model for the task so far
    if not is_sagemaker:
        # Update best_performance.json if performance improved
        best_performance_path = os.path.join(model_level_dir, "best_performance.json")
        
        try:
            # Get current validation accuracy
            val_accuracy = float(history.iloc[best_epoch-1]['Val Accuracy'])
            
            # Create dictionaries for test accuracies and F1 scores
            test_accuracies = {}
            test_f1_scores = {}
            for split_name, metrics in all_results.items():
                test_accuracies[split_name] = metrics['accuracy']
                test_f1_scores[split_name] = metrics['f1_score']
            
            # Load current best performance 
            if os.path.exists(best_performance_path):
                with open(best_performance_path, 'r') as f:
                    best_performance = json.load(f)
                
                # Get current best validation accuracy
                current_best_val = best_performance.get('best_val_accuracy', 0.0)
                
                # Compare using validation accuracy
                if val_accuracy > current_best_val:
                    print(f"New best model! Validation accuracy: {val_accuracy:.4f} (previous best: {current_best_val:.4f})")
                    
                    # Create updated best performance
                    updated_performance = {
                        'best_val_accuracy': val_accuracy,
                        'best_val_loss': float(history.iloc[best_epoch-1]['Val Loss']),
                        'best_experiment_id': experiment_id,
                        'best_experiment_params': config,
                        'best_test_accuracies': test_accuracies,
                        'best_test_f1_scores': test_f1_scores,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save updated best performance
                    with open(best_performance_path, 'w') as f:
                        json.dump(updated_performance, f, indent=4)
                else:
                    print(f"Not the best model. Current validation accuracy: {val_accuracy:.4f} (best: {current_best_val:.4f})")
            else:
                # First time creating the file
                print(f"Creating initial best performance record with validation accuracy: {val_accuracy:.4f}")
                
                initial_performance = {
                    'best_val_accuracy': val_accuracy,
                    'best_val_loss': float(history.iloc[best_epoch-1]['Val Loss']),
                    'best_experiment_id': experiment_id,
                    'best_experiment_params': config,
                    'best_test_accuracies': test_accuracies,
                    'best_test_f1_scores': test_f1_scores,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Create directory if needed
                os.makedirs(os.path.dirname(best_performance_path), exist_ok=True)
                
                # Save initial best performance
                with open(best_performance_path, 'w') as f:
                    json.dump(initial_performance, f, indent=4)
        except Exception as e:
            print(f"Warning: Failed to update best_performance.json: {e}")
            import traceback
            traceback.print_exc()
    
    return summary, all_results, model

if __name__ == '__main__':
    import math  # Import math here for the scheduler function
    main()
