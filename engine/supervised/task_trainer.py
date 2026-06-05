import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from engine.base_trainer import BaseTrainer
from tqdm import tqdm

def warmup_schedule(epoch, warmup_epochs):
    """Warmup learning rate schedule."""
    if epoch < warmup_epochs:
        # Linear warmup
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        # Cosine annealing
        return 0.5 * (1.0 + np.cos(np.pi * epoch / warmup_epochs))

class TaskTrainer(BaseTrainer):
    """Trainer for supervised learning tasks with CSI data."""
    
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, criterion=None, optimizer=None, 
                 scheduler=None, device='cuda:0', save_path='./results', checkpoint_path=None, 
                 num_classes=None, label_mapper=None, config=None, distributed=False, local_rank=0):
        """
        Initialize the task trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            save_path: Path to save results
            checkpoint_path: Path to load checkpoint
            num_classes: Number of classes for the model
            label_mapper: LabelMapper for mapping between class indices and names
            config: Configuration object with training parameters
            distributed: Whether this is a distributed training run
            local_rank: Local rank of this process in distributed training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.label_mapper = label_mapper
        self.config = config
        self.distributed = distributed
        self.local_rank = local_rank
        
        # Create directory if it doesn't exist
        if not distributed or (distributed and local_rank == 0):
            os.makedirs(save_path, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Load checkpoint if specified
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Log
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # Training tracking
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
    
    def setup_scheduler(self):
        """Set up learning rate scheduler."""
        warmup_epochs = getattr(self.config, 'warmup_epochs', 5)
        lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Train the model."""
        if not self.distributed or (self.distributed and self.local_rank == 0):
            print('Starting supervised training phase...')
        
        # Records for tracking progress
        records = []
        
        # Set default configuration values if config is None
        if self.config is None:
            epochs = 30
            patience = 15
        else:
            # Number of epochs and patience from config
            # Handle both object attributes and dictionary keys
            if isinstance(self.config, dict):
                epochs = self.config.get('epochs', 30)
                patience = self.config.get('patience', 15)
            else:
                epochs = getattr(self.config, 'epochs', 30)
                patience = getattr(self.config, 'patience', 15)
        
        # Best model state
        best_model = None
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Only print from rank 0 in distributed mode
            if not self.distributed or (self.distributed and self.local_rank == 0):
                print(f'Epoch {epoch+1}/{epochs}')
            
            # Set epoch for distributed sampler
            if self.distributed and hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train one epoch
            train_loss, train_acc, train_time = self.train_epoch()
            
            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # Update records
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Step scheduler
            self.scheduler.step()
            
            # Only log from rank 0 in distributed mode
            if not self.distributed or (self.distributed and self.local_rank == 0):
                # Record for this epoch
                record = {
                    'Epoch': epoch + 1,
                    'Train Loss': train_loss,
                    'Val Loss': val_loss,
                    'Train Accuracy': train_acc,
                    'Val Accuracy': val_acc,
                    'Time per sample': train_time
                }
                records.append(record)
                
                print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
                print(f'Time per sample: {train_time:.6f} seconds')
            
            # Early stopping check
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                
                # Save the best model - only from rank 0 in distributed mode
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    best_model_path = os.path.join(self.save_path, "best_model.pt")
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'epoch': epoch,
                    }, best_model_path)
                    print(f"Best model saved to {best_model_path}")
                
                self.best_epoch = epoch + 1
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    if not self.distributed or (self.distributed and self.local_rank == 0):
                        print(f'Early stopping triggered after {patience} epochs without improvement.')
                    self.model.load_state_dict(best_model)
                    break
        
        # Only perform final steps from rank 0 in distributed mode
        if not self.distributed or (self.distributed and self.local_rank == 0):
            # Create results DataFrame
            results_df = pd.DataFrame(records)
            
            # Save results
            results_df.to_csv(os.path.join(self.save_path, 'training_results.csv'), index=False)
            
            # Plot results
            self.plot_training_results()
        
        # Get the best validation accuracy and its corresponding epoch
        if len(self.val_accuracies) > 0:
            best_idx = np.argmax(self.val_accuracies)
            best_epoch = best_idx + 1
            best_val_accuracy = self.val_accuracies[best_idx]
        else:
            best_epoch = epochs
            best_val_accuracy = 0.0
        
        # Create a dictionary with unified information as the return value
        training_results = {
            'train_loss_history': self.train_losses,
            'val_loss_history': self.val_losses,
            'train_accuracy_history': self.train_accuracies,
            'val_accuracy_history': self.val_accuracies,
            'best_epoch': self.best_epoch,
            'best_val_accuracy': best_val_accuracy
        }
        
        # Only include DataFrame in non-distributed mode or from rank 0
        if not self.distributed or (self.distributed and self.local_rank == 0):
            training_results['training_dataframe'] = results_df
        
        return self.model, training_results
    
    def train_epoch(self):
        """Train the model for a single epoch.
        
        Returns:
            A tuple of (loss, accuracy, time_per_sample).
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        total_time = 0.0
        
        for inputs, labels in self.train_loader:
            # Skip empty batches (from custom_collate_fn if all samples were None)
            if inputs.size(0) == 0:
                continue
                
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Transfer to device
            inputs = inputs.to(self.device)
            
            # Handle case where labels might be a tuple
            if isinstance(labels, tuple):
                labels = labels[0]
            
            # Create batch of labels
            batch_size = inputs.size(0)
            
            # Handle case where labels might be strings or scalars
            if isinstance(labels, str):
                try:
                    # Create a tensor of the same value repeated batch_size times
                    label_value = int(labels)
                    labels = torch.tensor([label_value] * batch_size).to(self.device)
                except:
                    labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            elif not hasattr(labels, 'shape') or len(labels.shape) == 0:
                # Handle scalar labels by repeating them
                label_value = int(labels)
                labels = torch.tensor([label_value] * batch_size).to(self.device)
            else:
                # If it's already a batch, just move to device
                labels = labels.to(self.device)
            
            # One-hot encoding for labels if needed
            if self.criterion.__class__.__name__ in ['BCELoss', 'BCEWithLogitsLoss']:
                labels_one_hot = F.one_hot(labels, self.num_classes).float()
            else:
                labels_one_hot = labels
            
            start_time = time.time()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels_one_hot)
            
            # Backward pass
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Measure elapsed time
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Accumulate loss and accuracy
            epoch_loss += loss.item() * batch_size
            
            # Calculate accuracy
            if outputs.shape[1] > 1:  # Multi-class
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == labels).sum().item()
            else:  # Binary
                predicted = (outputs > 0.5).float()
                correct = (predicted == labels).sum().item()
                
            epoch_accuracy += correct
        
        # Calculate averages.  Guard against empty loaders.
        if total_samples == 0:
            print("[train_epoch] No samples in train loader; skipping averaging.")
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            time_per_sample = 0.0
        else:
            epoch_loss /= total_samples
            epoch_accuracy /= total_samples
            time_per_sample = total_time / total_samples
        
        # Synchronize metrics across processes in distributed training
        if self.distributed and torch.distributed.is_initialized():
            # Create tensors for each metric
            metrics = torch.tensor([epoch_loss, epoch_accuracy, time_per_sample, total_samples], 
                                  dtype=torch.float, device=self.device)
            
            # All-reduce to compute mean across processes
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            
            # Get world size for averaging
            world_size = torch.distributed.get_world_size()
            metrics /= world_size
            
            # Extract metrics
            epoch_loss = metrics[0].item()
            epoch_accuracy = metrics[1].item()
            time_per_sample = metrics[2].item()
        
        return epoch_loss, epoch_accuracy, time_per_sample
    
    def evaluate(self, data_loader):
        """Evaluate the model.
        
        Args:
            data_loader: The data loader to use for evaluation.
            
        Returns:
            A tuple of (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Skip empty batches
                if inputs.size(0) == 0:
                    continue
                    
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Transfer to device
                inputs = inputs.to(self.device)
                # Handle case where labels might be a tuple
                if isinstance(labels, tuple):
                    labels = labels[0]
                
                # Create batch of labels
                batch_size = inputs.size(0)
                
                # Handle case where labels might be strings or scalars
                if isinstance(labels, str):
                    try:
                        # Create a tensor of the same value repeated batch_size times
                        label_value = int(labels)
                        labels = torch.tensor([label_value] * batch_size).to(self.device)
                    except:
                        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                elif not hasattr(labels, 'shape') or len(labels.shape) == 0:
                    # Handle scalar labels by repeating them
                    label_value = int(labels)
                    labels = torch.tensor([label_value] * batch_size).to(self.device)
                else:
                    # If it's already a batch, just move to device
                    labels = labels.to(self.device)
                
                # One-hot encoding for labels if needed
                if self.criterion.__class__.__name__ in ['BCELoss', 'BCEWithLogitsLoss']:
                    labels_one_hot = F.one_hot(labels, self.num_classes).float()
                else:
                    labels_one_hot = labels
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_one_hot)
                
                # Accumulate loss
                total_loss += loss.item() * batch_size
                
                # Calculate accuracy
                if outputs.shape[1] > 1:  # Multi-class
                    predicted = torch.argmax(outputs, dim=1)
                    correct = (predicted == labels).sum().item()
                else:  # Binary
                    predicted = (outputs > 0.5).float()
                    correct = (predicted == labels).sum().item()
                    
                total_correct += correct
        
        # Calculate averages.  Guard against empty loaders (e.g. a difficulty
        # tier with zero samples) so we surface a 0/0 result instead of crashing.
        if total_samples == 0:
            print("[evaluate] No samples in this loader; returning zeros.")
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
        
        # Synchronize metrics across processes in distributed training
        if self.distributed and torch.distributed.is_initialized():
            # Create tensors for each metric
            metrics = torch.tensor([avg_loss, accuracy, total_samples], 
                                 dtype=torch.float, device=self.device)
            
            # All-reduce to compute mean across processes
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            
            # Get world size for averaging
            world_size = torch.distributed.get_world_size()
            
            # For loss and accuracy we want the average, but we need to account for the
            # different number of samples each process may have processed
            world_samples = metrics[2].item()
            if world_samples > 0:
                avg_loss = metrics[0].item() * world_size / world_samples
                accuracy = metrics[1].item() * world_size / world_samples
        
        return avg_loss, accuracy
    
    def plot_training_results(self):
        """Plot the training results."""
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].grid(True)
        
        # Plot validation loss
        axs[0, 1].plot(self.val_losses)
        axs[0, 1].set_title('Validation Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
        
        # Plot training accuracy
        axs[1, 0].plot(self.train_accuracies)
        axs[1, 0].set_title('Training Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].grid(True)
        
        # Plot validation accuracy
        axs[1, 1].plot(self.val_accuracies)
        axs[1, 1].set_title('Validation Accuracy')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_results.png'))
        plt.close()
        
        # Also plot confusion matrix
        self.plot_confusion_matrix()
    
    def plot_confusion_matrix(self, data_loader=None, epoch=None, mode='val'):
        """
        Plot the confusion matrix and save the figure.
        
        Args:
            data_loader: Dataloader to use for evaluation
            epoch: Current epoch
            mode: 'val' or 'test' mode
        """
        # Set evaluation mode
        self.model.eval()
        
        # Use validation loader if not specified
        if data_loader is None:
            if mode == 'val' and self.val_loader is not None:
                data_loader = self.val_loader
            elif mode == 'test' and self.test_loader is not None:
                data_loader = self.test_loader
            else:
                raise ValueError(f"No data loader available for mode {mode}")
        
        # Collect all predictions and labels
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get data and labels
                if isinstance(batch, dict):
                    data = batch['data']
                    labels = batch['labels']
                else:
                    data, labels = batch
                
                # Handle different label formats
                if isinstance(labels, tuple):
                    # Use the first element as class label
                    labels = labels[0]
                
                # Move data to device
                data = data.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                elif isinstance(labels, (list, np.ndarray)):
                    labels = torch.tensor(labels).to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Get class names if available
        class_names = None
        if self.label_mapper is not None:
            class_names = [self.label_mapper.get_name(i) for i in range(self.num_classes)]
        
        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix ({mode})')
        
        # Save figure
        epoch_str = f'_epoch{epoch}' if epoch is not None else ''
        plt.savefig(os.path.join(self.save_path, f'confusion_matrix_{mode}{epoch_str}.png'))
        plt.close()
        
        # Generate and save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Replace indices with class names if available
        if class_names is not None:
            # Create a mapping dictionary from indices to class names
            index_to_name = {}
            for i, name in enumerate(class_names):
                index_to_name[str(i)] = name
            
            # Replace indices with class names
            new_index = []
            for idx in report_df.index:
                if idx in index_to_name:
                    new_index.append(index_to_name[idx])
                else:
                    new_index.append(idx)
            
            report_df.index = new_index
        
        # Save report
        report_df.to_csv(os.path.join(self.save_path, f'classification_report_{mode}{epoch_str}.csv'))
        
        return report_df

    def calculate_metrics(self, data_loader, epoch=None):
        """
        Calculate overall performance metrics, including weighted F1 score.
        
        Args:
            data_loader: Data loader for evaluation
            epoch: Current epoch (optional)
            
        Returns:
            Tuple of (weighted_f1_score, per_class_f1_scores)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize lists to store predictions and ground truth
        all_preds = []
        all_targets = []
        
        # No gradient during evaluation
        with torch.no_grad():
            for batch in data_loader:
                # Get data and move to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, targets = batch
                else:
                    # Handle case where batch is a dictionary
                    data = batch['input']
                    targets = batch['target']
                
                # Skip empty batches
                if data.size(0) == 0:
                    continue
                
                # Move to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Append to lists
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Validate data before calculating metrics
        if len(all_preds) == 0 or len(all_targets) == 0:
            print(f"Warning: Empty prediction or target arrays, skipping F1 calculation")
            return 0.0, pd.DataFrame()
            
        if len(all_preds) != len(all_targets):
            print(f"Warning: Prediction and target array lengths don't match: {len(all_preds)} vs {len(all_targets)}")
            return 0.0, pd.DataFrame()
            
        # Print some debug information
        print(f"Predictions shape: {all_preds.shape}, unique values: {np.unique(all_preds)}")
        print(f"Targets shape: {all_targets.shape}, unique values: {np.unique(all_targets)}")
        
        # Calculate weighted F1 score
        from sklearn.metrics import f1_score, classification_report
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        
        # Calculate per-class F1 scores
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        
        # Get detailed classification report
        report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
        
        # Save the report to a CSV file if epoch is None (final evaluation)
        if epoch is None and hasattr(self, 'save_path'):
            import pandas as pd
            # Convert report to DataFrame
            report_df = pd.DataFrame(report).transpose()
            
            # Determine split name from data_loader (assuming it's in the dataloader's dataset attributes)
            split_name = getattr(data_loader.dataset, 'split', 'unknown')
            
            # Save to CSV
            report_path = os.path.join(self.save_path, f'classification_report_{split_name}.csv')
            report_df.to_csv(report_path)
            print(f"Classification report saved to {report_path}")
            
            return weighted_f1, report_df
        
        return weighted_f1, pd.DataFrame(report).transpose()

    def training_loop(self, base_lr=1e-3, weight_decay=0.0, clip_grad=None):
        
        # Get model, loaders, criterion, etc
        model = self.model
        train_loader = self.train_loader
        val_loader = self.val_loader
        criterion = self.criterion
        device = self.device
        
        # Setup optimizer
        optimizer = self.optimizer or torch.optim.Adam(
            model.parameters(), 
            lr=base_lr, 
            weight_decay=weight_decay
        )
        
        # Setup learning rate scheduler
        if self.scheduler is None:
            if self.config is None:
                # No config, use default values
                patience = 7
                epochs = 30
                warmup_epochs = 5
            else:
                # Use config
                if isinstance(self.config, dict):
                    patience = self.config.get('patience', 7)
                    epochs = self.config.get('epochs', 30)
                    warmup_epochs = self.config.get('warmup_epochs', 5)
                else:
                    patience = getattr(self.config, 'patience', 7)
                    epochs = getattr(self.config, 'epochs', 30)
                    warmup_epochs = getattr(self.config, 'warmup_epochs', 5)
                
            # Create scheduler
            self.scheduler = WarmupCosineScheduler(
                optimizer, 
                epochs, 
                warmup_epochs=warmup_epochs, 
                min_lr=1e-6
            )
        
        scheduler = self.scheduler
        
        # Training loop
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_model_state = None
        no_improve = 0
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            print(f'Epoch {epoch+1}/{epochs}')
            pbar = tqdm(train_loader)
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                train_correct += batch_correct
                train_total += targets.size(0)
                
                # Update progress bar with current statistics
                pbar.set_description(f'Train Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*train_correct/train_total:.2f}%')
            
            # Step scheduler
            scheduler.step()
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(targets).sum().item()
                    val_total += targets.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            
            # Save metrics to history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
            print(f'  Learning Rate: {scheduler.get_lr()[0]:.6f}')
            
            # Create confusion matrix for validation set
            y_preds, y_true = self.get_predictions(val_loader)
            # No need to generate confusion matrix here if we only want to save the best model's
            
            # Save the best model
            is_best = False
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch
                no_improve = 0
                is_best = True
                
                # Save current best model
                self.save_checkpoint(epoch, model, optimizer, scheduler, is_best)
            else:
                no_improve += 1
                
            # Early stopping
            if no_improve >= patience:
                print(f'Early stopping after {epoch+1} epochs without improvement')
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # At the end, generate and save confusion matrix for the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f'Generating confusion matrix for best model (epoch {best_epoch+1})')
            self.plot_confusion_matrix(epoch=best_epoch+1, mode='val_best')
        
        # If we didn't complete all epochs, set best_epoch to the last epoch
        if 'best_epoch' not in locals():
            best_epoch = epochs
        
        return history, best_val_loss, best_val_acc, best_model_state, best_epoch
