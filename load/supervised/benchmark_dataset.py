import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from scipy.io import loadmat
from pathlib import Path
import scipy.io as sio
from ..supervised.label_utils import LabelMapper, create_label_mapper_from_metadata

class BenchmarkCSIDataset(Dataset):
    """
    Dataset for WiFi CSI benchmark with split handling and metadata loading.
    Supports MAT, NPY, and HDF5 formats.
    Each sample has shape (time_index, feature_size, 1) in H5 files.
    """
    def __init__(self, 
                 dataset_root,  # Root directory of the dataset
                 task_name,     # Name of the task (e.g., 'motion_source_recognition')
                 split_name,    # Split name (e.g., 'train_id', 'test_cross_user')
                 transform=None,
                 target_transform=None,
                 file_format="h5",  # "mat", "npy", or "h5"
                 data_column="file_path",  # Column in metadata that points to data
                 label_column="label",   # Column in metadata for label
                 data_key="CSI_amps",   # Key in h5 file for data
                 label_mapper=None,     # Optional label mapper for converting string labels to indices
                 task_dir=None,        # Optional task directory path (to avoid searching again)
                 debug=False):         # Flag to enable/disable debug print statements
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.file_format = file_format.lower()
        self.data_column = data_column
        self.label_column = label_column
        self.data_key = data_key
        self.debug = debug
        
        # Use provided task_dir if available
        if task_dir is not None and os.path.isdir(task_dir):
            self.task_dir = task_dir
            print(f"Using provided task directory: {self.task_dir}")
        else:
            # Build paths
            possible_paths = [
                os.path.join(dataset_root, "tasks", task_name),              # dataset_root/tasks/task_name
                os.path.join(dataset_root, task_name),                        # dataset_root/task_name
                os.path.join(dataset_root, task_name.lower()),                # dataset_root/task_name_lowercase
                os.path.join(dataset_root, "tasks", task_name.lower())        # dataset_root/tasks/task_name_lowercase
            ]
            
            task_dir_found = None
            for path in possible_paths:
                print(f"Dataset checking path: {path}")
                if os.path.isdir(path):
                    # Check if this directory has metadata and splits
                    has_metadata = os.path.exists(os.path.join(path, 'metadata'))
                    has_splits = os.path.exists(os.path.join(path, 'splits'))
                    print(f"  Has metadata: {has_metadata}, Has splits: {has_splits}")
                    
                    if has_metadata or has_splits:
                        task_dir_found = path
                        break
            
            # If not found, try walking the directory to find it
            if task_dir_found is None:
                print(f"Dataset task directory not found in predefined paths, searching recursively...")
                for root, dirs, files in os.walk(dataset_root):
                    if task_name in dirs or task_name.lower() in dirs:
                        # Try with exact case first
                        if task_name in dirs:
                            potential_task_dir = os.path.join(root, task_name)
                        else:
                            potential_task_dir = os.path.join(root, task_name.lower())
                        
                        # Check if this directory has metadata or splits
                        has_metadata = os.path.exists(os.path.join(potential_task_dir, 'metadata'))
                        has_splits = os.path.exists(os.path.join(potential_task_dir, 'splits'))
                        print(f"Found potential directory: {potential_task_dir}")
                        print(f"  Has metadata: {has_metadata}, Has splits: {has_splits}")
                        
                        if has_metadata or has_splits:
                            task_dir_found = potential_task_dir
                            break
            
            if task_dir_found is None:
                raise ValueError(f"Could not find task directory for {task_name} in {dataset_root}")
            
            self.task_dir = task_dir_found
            print(f"Found task directory: {self.task_dir}")
        
        # Now use the task_dir to find the split and metadata files
        split_path = os.path.join(self.task_dir, "splits", f"{split_name}.json")
        metadata_path = os.path.join(self.task_dir, "metadata", "sample_metadata.csv")
        
        # Check if files exist
        if not os.path.exists(split_path):
            # Try alternate locations for split file
            alternate_split_paths = [
                os.path.join(self.task_dir, f"splits/{split_name}.json"),
                os.path.join(self.task_dir, f"{split_name}.json"),
                os.path.join(dataset_root, f"splits/{split_name}.json")
            ]
            
            found_split = False
            for alt_path in alternate_split_paths:
                if os.path.exists(alt_path):
                    split_path = alt_path
                    found_split = True
                    break
                    
            if not found_split:
                # List available splits
                splits_dir = os.path.join(self.task_dir, "splits")
                available_splits = []
                if os.path.exists(splits_dir) and os.path.isdir(splits_dir):
                    for file in os.listdir(splits_dir):
                        if file.endswith('.json'):
                            available_splits.append(file.replace('.json', ''))
                
                raise FileNotFoundError(f"Split file not found: {split_path}\nAvailable splits: {available_splits}")
        
        if not os.path.exists(metadata_path):
            # Try alternate locations for metadata file
            alternate_metadata_paths = [
                os.path.join(self.task_dir, "metadata", "metadata.csv"),
                os.path.join(self.task_dir, "metadata", "sample_metadata.csv"),
                os.path.join(self.task_dir, "subset_metadata.csv"),
                os.path.join(self.task_dir, "metadata.csv")
            ]
            
            found_metadata = False
            for alt_path in alternate_metadata_paths:
                if os.path.exists(alt_path):
                    metadata_path = alt_path
                    found_metadata = True
                    break
                    
            if not found_metadata:
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load split IDs
        with open(split_path, 'r') as f:
            self.split_ids = set(json.load(f))
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter metadata for this split
        id_column = 'id' if 'id' in self.metadata.columns else 'sample_id'
        self.split_metadata = self.metadata[self.metadata[id_column].isin(self.split_ids)].reset_index(drop=True)
        
        # Check if we have the required columns
        if data_column not in self.split_metadata.columns:
            raise ValueError(f"Data column '{data_column}' not found in metadata")
        if label_column not in self.split_metadata.columns:
            raise ValueError(f"Label column '{label_column}' not found in metadata")
            
        print(f"Loaded {len(self.split_metadata)} samples for {task_name} - {split_name}")
        
        # Initialize label mapper if not provided
        if label_mapper is None:
            # Use the task_dir
            mapping_path = os.path.join(self.task_dir, 'metadata', 'label_mapping.json')
            
            # Create mapper directory if it doesn't exist
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            
            if os.path.exists(mapping_path):
                # Load existing mapping
                self.label_mapper = LabelMapper.load(mapping_path)
                print(f"Loaded existing label mapping with {self.label_mapper.num_classes} classes")
            else:
                # Create new mapping
                self.label_mapper, _ = create_label_mapper_from_metadata(
                    metadata_path, 
                    label_column=label_column,
                    save_path=mapping_path
                )
        else:
            self.label_mapper = label_mapper
        
        # Set number of classes
        self.num_classes = self.label_mapper.num_classes
    
    def __len__(self):
        return len(self.split_metadata)
    
    def __getitem__(self, idx):
        """Get sample by index"""
        try:
            row = self.split_metadata.iloc[idx]
            
            # Get file path (might be relative to dataset_root)
            original_filepath = row[self.data_column]
            filepath = original_filepath
            
            # Check if task_dir is the same as dataset_root (use_root_as_task_dir=True case)
            using_root_as_task_dir = os.path.normpath(self.task_dir) == os.path.normpath(self.dataset_root)
            
            # Handle the paths from the metadata file
            if os.path.exists(original_filepath):
                # If the original path exists as-is, use it directly
                filepath = original_filepath
            elif original_filepath.startswith('E:/'):
                # Handle paths that start with the specific Windows drive path
                # Extract the relative path after potential prefixes
                relative_parts = []
                path_parts = original_filepath.replace('\\', '/').split('/')
                
                # Try to find the task name in the path to extract relevant parts
                for i, part in enumerate(path_parts):
                    if part.lower() == self.task_name.lower() or part.lower() == self.task_name.replace('_', '').lower():
                        relative_parts = path_parts[i:]
                        break
                
                if relative_parts:
                    # Construct path based on whether we're using root as task dir
                    if using_root_as_task_dir:
                        filepath = os.path.join(self.dataset_root, *relative_parts)
                    else:
                        filepath = os.path.join(self.dataset_root, 'tasks', *relative_parts)
                else:
                    # Try with the last parts of the path (after the last 'CSI100Hz' if it exists)
                    for i, part in enumerate(path_parts):
                        if part == 'CSI100Hz':
                            relative_parts = path_parts[i+1:]
                            break
                    
                    if relative_parts:
                        if using_root_as_task_dir:
                            filepath = os.path.join(self.dataset_root, *relative_parts)
                        else:
                            filepath = os.path.join(self.dataset_root, 'tasks', self.task_name, *relative_parts)
                    else:
                        # Last resort: try to find path segments like sub_X/user_Y/act_Z
                        try_sub_user_path = False
                        for i, part in enumerate(path_parts):
                            if part.startswith('sub_') and i+1 < len(path_parts) and path_parts[i+1].startswith('user_'):
                                relative_parts = path_parts[i:]
                                try_sub_user_path = True
                                break
                        
                        if try_sub_user_path:
                            if using_root_as_task_dir:
                                filepath = os.path.join(self.dataset_root, *relative_parts)
                            else:
                                filepath = os.path.join(self.dataset_root, 'tasks', self.task_name, *relative_parts)
                        else:
                            # Try just the filename
                            filename = os.path.basename(original_filepath)
                            if using_root_as_task_dir:
                                filepath = os.path.join(self.dataset_root, filename)
                            else:
                                filepath = os.path.join(self.dataset_root, 'tasks', self.task_name, filename)
            elif not os.path.isabs(filepath):
                # Handle relative paths based on whether we're using root as task dir
                if using_root_as_task_dir:
                    # When using root as task dir, use paths directly
                    # Case 1: Path includes 'tasks/TaskName/...'
                    if filepath.startswith('/tasks/') or filepath.startswith('tasks/') or filepath.startswith('tasks\\'):
                        filepath = os.path.join(self.dataset_root, filepath)
                    # Case 2: Path is just 'TaskName/...' or contains TaskName directory
                    elif filepath.startswith(f"{self.task_name}/") or filepath.startswith(f"{self.task_name}\\"):
                        # In this case, we might need to omit the task name in the path
                        # Extract the path parts after the task name
                        path_parts = filepath.replace('\\', '/').split('/')
                        if path_parts[0] == self.task_name:
                            filepath = os.path.join(self.dataset_root, *path_parts[1:])
                        else:
                            filepath = os.path.join(self.dataset_root, filepath)
                    # Case 3: Path is relative to dataset root
                    else:
                        filepath = os.path.join(self.dataset_root, filepath)
                else:
                    # Original behavior when not using root as task dir
                    # Case 1: Path includes 'tasks/TaskName/...'
                    if filepath.startswith('/tasks/') or filepath.startswith('tasks/') or filepath.startswith('tasks\\'):
                        filepath = os.path.join(self.dataset_root, filepath)
                    # Case 2: Path is just 'TaskName/...'
                    elif filepath.startswith(f"{self.task_name}/") or filepath.startswith(f"{self.task_name}\\"):
                        filepath = os.path.join(self.dataset_root, 'tasks', filepath)
                    # Case 3: Path is relative to task directory
                    else:
                        filepath = os.path.join(self.dataset_root, 'tasks', self.task_name, filepath)
            
            # Check if file exists
            if not os.path.exists(filepath):
                # Try alternative constructions if file not found
                alt_paths = []
                
                # Alternative 1: Try joining directly
                alt1 = os.path.join(self.dataset_root, original_filepath)
                alt_paths.append(("Direct join", alt1))
                
                # Alternative 2: Try with 'tasks' prefix if not using root as task dir
                if not using_root_as_task_dir and 'tasks' not in original_filepath:
                    alt2 = os.path.join(self.dataset_root, 'tasks', original_filepath)
                    alt_paths.append(("With tasks prefix", alt2))
                
                # Alternative 3: Try with task name if not using root as task dir
                if not using_root_as_task_dir and self.task_name not in original_filepath:
                    alt3 = os.path.join(self.dataset_root, 'tasks', self.task_name, original_filepath)
                    alt_paths.append(("Using path segments", alt3))
                
                # Alternative 4: Try without tasks directory when using root as task dir
                if using_root_as_task_dir:
                    # Look for pattern: /tasks/TaskName/... in filepath and remove those parts
                    filepath_parts = filepath.replace('\\', '/').split('/')
                    if 'tasks' in filepath_parts and self.task_name in filepath_parts:
                        # Find the position of task_name in the path
                        try:
                            task_idx = filepath_parts.index(self.task_name)
                            # If 'tasks' comes right before task_name, eliminate both
                            if task_idx > 0 and filepath_parts[task_idx-1] == 'tasks':
                                # Reconstruct path without 'tasks/task_name'
                                alt4 = os.path.join(self.dataset_root, *filepath_parts[task_idx+1:])
                                alt_paths.append(("Removing tasks/TaskName prefix", alt4))
                        except ValueError:
                            pass
                
                # Check alternatives
                for desc, alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        if self.debug:
                            print(f"Found file using alternative path ({desc}): {alt_path}")
                        filepath = alt_path
                        break
                
                # If still not found, raise error with detailed information
                if not os.path.exists(filepath):
                    error_msg = f"File not found: {filepath}\n"
                    error_msg += f"Original path from CSV: {original_filepath}\n"
                    error_msg += f"Dataset root: {self.dataset_root}\n"
                    error_msg += f"Task directory: {self.task_dir}\n"
                    error_msg += f"Using root as task dir: {using_root_as_task_dir}\n"
                    error_msg += "Tried alternative paths:\n"
                    for desc, alt_path in alt_paths:
                        error_msg += f"  - {desc}: {alt_path}\n"
                    
                    # Additional debug info about directory structure
                    error_msg += "\nDirectory structure at dataset root:\n"
                    try:
                        for item in os.listdir(self.dataset_root):
                            item_path = os.path.join(self.dataset_root, item)
                            if os.path.isdir(item_path):
                                error_msg += f"  Directory: {item}/\n"
                                # List first few items in subdirectory
                                try:
                                    sub_items = os.listdir(item_path)[:5]  # First 5 items
                                    for sub_item in sub_items:
                                        error_msg += f"    - {sub_item}\n"
                                    if len(os.listdir(item_path)) > 5:
                                        error_msg += f"    - ... ({len(os.listdir(item_path))-5} more items)\n"
                                except Exception as e:
                                    error_msg += f"    - Error listing contents: {e}\n"
                            else:
                                error_msg += f"  File: {item}\n"
                    except Exception as e:
                        error_msg += f"  Error listing directory: {e}\n"
                    
                    raise FileNotFoundError(error_msg)
            
            # Load data based on file format
            if self.file_format == "npy":
                try:
                    csi_data = np.load(filepath)
                except Exception as e:
                    print(f"Skipping file {filepath} due to error: {str(e)}")
                    return None
            elif self.file_format == "mat":
                try:
                    mat_dict = loadmat(filepath)
                    csi_data = self._extract_csi_from_mat(mat_dict)
                    if csi_data is None:
                        print(f"Skipping file {filepath} due to error: Could not extract CSI data from MAT file")
                        return None
                except Exception as e:
                    print(f"Skipping file {filepath} due to error: {str(e)}")
                    return None
            elif self.file_format == "h5":
                try:
                    with h5py.File(filepath, 'r') as f:
                        if len(f.keys()) == 0:
                            print(f"Skipping file {filepath} due to error: 'Empty H5 file: {filepath}'")
                            return None
                        
                        # Use the data_key (default is 'CSI_amps') instead of hardcoded 'csi'
                        if self.data_key in f:
                            csi_data = np.array(f[self.data_key])
                        else:
                            # Fallback to checking other common keys
                            if 'csi' in f:
                                csi_data = np.array(f['csi'])
                            elif 'CSI' in f:
                                csi_data = np.array(f['CSI'])
                            else:
                                print(f"Skipping file {filepath} due to error: Could not find data in H5 file. Available keys: {list(f.keys())}")
                                return None
                except Exception as e:
                    print(f"Skipping file {filepath} due to error: '{str(e)}'")
                    return None
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
            
            # Check if data is valid
            if csi_data is None or csi_data.size == 0:
                print(f"Skipping file {filepath} due to error: Empty data")
                return None
            
            # Convert to tensor
            csi_data = torch.from_numpy(csi_data).float()
            
            # Reshape to (1, time_index, feature_size)
            if len(csi_data.shape) == 3:  # (time_index, feature_size, 1)
                # Permute to get (1, time_index, feature_size)
                csi_data = csi_data.permute(2, 1, 0)
            
            # Normalize CSI data along time and feature dimensions
            # First calculate mean and std along time and feature dimensions (dims 1 and 2)
            mean = csi_data.mean(dim=(1, 2), keepdim=True)
            std = csi_data.std(dim=(1, 2), keepdim=True)
            # Add small epsilon to avoid division by zero
            std = torch.clamp(std, min=1e-8)
            # Apply normalization
            csi_data = (csi_data - mean) / std
            
            # Standardize the size to (1, 500, 232)
            batch_size, time_index, feature_size = csi_data.shape
            target_time_index, target_feature_size = 500, 232
            
            # Create a tensor of zeros with the target shape
            standardized_data = torch.zeros((batch_size, target_time_index, target_feature_size), dtype=csi_data.dtype)
            
            # Calculate dimensions for copying (clip or use the smaller of original and target)
            copy_time = min(time_index, target_time_index)
            copy_feature = min(feature_size, target_feature_size)
            
            # Copy data to the standardized tensor (handles both clipping and partial filling)
            standardized_data[:, :copy_time, :copy_feature] = csi_data[:, :copy_time, :copy_feature]
            
            # Update the tensor reference
            csi_data = standardized_data
            
            # Apply transforms
            if self.transform:
                csi_data = self.transform(csi_data)
            
            # Get label
            label = row[self.label_column]
            if self.target_transform:
                label = self.target_transform(label)
            
            # Convert string label to integer using mapper
            label_idx = self.label_mapper.transform(label)
            
            return csi_data, label_idx
            
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            return None
    
    def _extract_csi_from_mat(self, mat_dict):
        """Extract CSI data from MAT file dict"""
        # Adjust this based on your MAT file structure
        if 'csi_data' in mat_dict:
            return mat_dict['csi_data']
        elif 'csi' in mat_dict:
            return mat_dict['csi']
        elif 'CSI_amps' in mat_dict:
            return mat_dict['CSI_amps']
        else:
            # Try to find a likely CSI array
            largest_array = None
            largest_size = 0
            for key, value in mat_dict.items():
                if isinstance(value, np.ndarray) and value.size > largest_size:
                    largest_array = value
                    largest_size = value.size
            return largest_array

    def get_label_counts(self):
        """Return a dictionary of label counts"""
        return self.split_metadata[self.label_column].value_counts().to_dict()
        
    def get_label_names(self):
        """Return the unique label names"""
        return self.split_metadata[self.label_column].unique().tolist()
