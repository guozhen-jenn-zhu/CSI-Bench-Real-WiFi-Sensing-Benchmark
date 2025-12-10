import os
from torch.utils.data import DataLoader
from .benchmark_dataset import BenchmarkCSIDataset
from ..supervised.label_utils import LabelMapper, create_label_mapper_from_metadata
import torch

def load_benchmark_supervised(
    dataset_root,
    task_name,
    batch_size=32,
    transform=None,
    target_transform=None,
    file_format="h5",
    data_column="file_path",
    label_column="label",
    data_key="CSI_amps",
    num_workers=4,
    shuffle_train=True,
    train_split="train_id",
    val_split="val_id",
    test_splits="all",
    use_root_as_task_dir=False,
    debug=False,
    distributed=False,
    collate_fn=None,
    pin_memory=True
):
    """
    Load benchmark dataset for supervised learning.
    
    Args:
        dataset_root: Root directory of the dataset.
        task_name: Name of the task (e.g., 'motion_source_recognition')
        batch_size: Batch size for DataLoader.
        transform: Optional transform to apply to data.
        target_transform: Optional transform to apply to labels.
        file_format: File format for data files ("h5", "mat", or "npy").
        data_column: Column in metadata that contains file paths.
        label_column: Column in metadata that contains labels.
        data_key: Key in h5 file for CSI data.
        num_workers: Number of worker processes for DataLoader.
        shuffle_train: Whether to shuffle training data.
        train_split: Name of training split.
        val_split: Name of validation split.
        test_splits: List of test split names or "all" to load all test_*.json splits.
        use_root_as_task_dir: Whether to directly use dataset_root as the task directory.
        debug: Whether to enable debug mode.
        distributed: Whether to configure data loaders for distributed training
        collate_fn: Custom collate function for DataLoader
        pin_memory: Whether to use pinned memory (set to False for MPS device)
        
    Returns:
        Dictionary with data loaders and number of classes.
    """
    # Set default test splits if not provided
    if test_splits is None:
        test_splits = ["test_id"]
    elif isinstance(test_splits, str):
        if test_splits.lower() == "all":
            # Handle the special 'all' case - will find all test_*.json files later
            test_splits = "all"
        else:
            test_splits = [test_splits]
    
    # Debug output
    data_dir_debug = os.path.join(dataset_root, "tasks", task_name)
    print(f"________________________DATA DEBUG________{data_dir_debug}")
    
    # If use_root_as_task_dir is specified
    if use_root_as_task_dir:
        # Check if root directory contains necessary subdirectories
        has_metadata = os.path.exists(os.path.join(dataset_root, 'metadata'))
        has_splits = os.path.exists(os.path.join(dataset_root, 'splits'))
        
        if has_metadata and has_splits:
            print(f"Using dataset root directly as task directory: {dataset_root}")
            task_dir = dataset_root
        else:
            print(f"Root directory does not contain required subfolders, will try standard paths.")
            use_root_as_task_dir = False  # Fall back to standard path search
    
    # If not using root directory directly or root directory doesn't meet requirements
    if not use_root_as_task_dir:
        # Try multiple directory structures to find the task directory
        possible_paths = [
            os.path.join(dataset_root, "tasks", task_name),              # dataset_root/tasks/task_name
            os.path.join(dataset_root, task_name),                        # dataset_root/task_name
            os.path.join(dataset_root, task_name.lower()),                # dataset_root/task_name_lowercase
            os.path.join(dataset_root, "tasks", task_name.lower())        # dataset_root/tasks/task_name_lowercase
        ]
        
        task_dir = None
        for path in possible_paths:
            print(f"Checking path: {path}")
            if os.path.isdir(path):
                # Check if this directory has metadata and splits
                has_metadata = os.path.exists(os.path.join(path, 'metadata'))
                has_splits = os.path.exists(os.path.join(path, 'splits'))
                print(f"  Has metadata: {has_metadata}, Has splits: {has_splits}")
                
                if has_metadata or has_splits:
                    task_dir = path
                    break
        
        # If not found, try walking the directory to find it
        if task_dir is None:
            print(f"Task directory not found in predefined paths, searching recursively...")
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
                        task_dir = potential_task_dir
                        break
    
    # Final check if task directory was found
    if task_dir is None:
        # If still no directory found and parameter allows, try using root directory again (even if missing standard subdirectories)
        if not use_root_as_task_dir and os.path.exists(dataset_root):
            print(f"No task directory found. As a last resort, trying to use root directory: {dataset_root}")
            task_dir = dataset_root
        else:
            raise ValueError(f"Could not find task directory for {task_name} in {dataset_root}")
    
    print(f"Using task directory: {task_dir}")
    
    # Display task directory contents for debugging
    print(f"\n==== Task Directory Contents ({task_dir}) ====")
    try:
        dir_items = os.listdir(task_dir)
        for item in dir_items:
            item_path = os.path.join(task_dir, item)
            if os.path.isdir(item_path):
                subdir_items = os.listdir(item_path)
                print(f"  Directory: {item}/ ({len(subdir_items)} items)")
                # If it's a metadata or splits directory, display its contents
                if item in ['metadata', 'splits']:
                    for subitem in subdir_items:
                        print(f"    - {subitem}")
            else:
                print(f"  File: {item}")
    except Exception as e:
        print(f"Cannot list task directory contents: {e}")
    print("================================================\n")
    
    # If test_splits is "all", find all test_*.json files in the splits directory
    if test_splits == "all":
        test_splits = []
        splits_dir = os.path.join(task_dir, 'splits')
        if os.path.exists(splits_dir) and os.path.isdir(splits_dir):
            for file in os.listdir(splits_dir):
                if file.startswith('test_') and file.endswith('.json'):
                    # Extract split name without .json extension
                    split_name = file[:-5]  # Remove .json extension
                    test_splits.append(split_name)
            print(f"Found {len(test_splits)} test splits: {test_splits}")
        
        # If no test splits found, default to test_id
        if not test_splits:
            test_splits = ["test_id"]
            print("No test splits found, defaulting to 'test_id'")
    
    # Create all split names
    all_splits = [train_split, val_split] + test_splits
    
    metadata_path = os.path.join(task_dir, 'metadata', 'sample_metadata.csv')
    mapping_path = os.path.join(task_dir, 'metadata', 'label_mapping.json')
    
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        # Try alternative metadata file names
        alternate_paths = [
            os.path.join(task_dir, 'metadata', 'metadata.csv'),
            os.path.join(task_dir, 'metadata', 'subset_metadata.csv'),
            os.path.join(task_dir, 'sample_metadata.csv'),  # Direct in task directory
            os.path.join(task_dir, 'subset_metadata.csv'),
            os.path.join(task_dir, 'metadata.csv')
        ]
        
        for alt_path in alternate_paths:
            print(f"Checking alternate metadata path: {alt_path}")
            if os.path.exists(alt_path):
                metadata_path = alt_path
                break
        
        if not os.path.exists(metadata_path):
            # Diagnostic information
            print(f"\n===== Metadata File Search Failed =====")
            print(f"Tried the following paths:")
            print(f"- {os.path.join(task_dir, 'metadata', 'sample_metadata.csv')}")
            for path in alternate_paths:
                print(f"- {path}")
            
            # Check if there are any .csv files in the task directory that might be metadata files with different names
            found_csvs = []
            for root, _, files in os.walk(task_dir):
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        found_csvs.append(csv_path)
                        print(f"Found possible CSV file: {csv_path}")
            
            # If any CSV files found, try using the first one
            if found_csvs:
                print(f"Trying to use first found CSV file: {found_csvs[0]}")
                metadata_path = found_csvs[0]
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Please make sure the data structure is correct and includes necessary metadata files.")
    
    print(f"Using metadata path: {metadata_path}")
    
    # Create or load label mapper
    if os.path.exists(mapping_path):
        label_mapper = LabelMapper.load(mapping_path)
    else:
        # Try to create the metadata directory if it doesn't exist
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        label_mapper, _ = create_label_mapper_from_metadata(
            metadata_path, 
            label_column=label_column,
            save_path=mapping_path
        )
    
    # Create datasets
    datasets = {}
    for split_name in all_splits:
        try:
            print(f"Using provided task directory: {task_dir}")
            dataset = BenchmarkCSIDataset(
                dataset_root=dataset_root,
                task_name=task_name,
                split_name=split_name,
                transform=transform,
                target_transform=target_transform,
                file_format=file_format,
                data_column=data_column,
                label_column=label_column,
                data_key=data_key,
                label_mapper=label_mapper,
                task_dir=task_dir,  # Pass the found task_dir to the dataset
                debug=debug
            )
            datasets[split_name] = dataset
            print(f"Loaded {len(dataset)} samples for {task_name} - {split_name}")
        except Exception as e:
            print(f"Error loading split '{split_name}': {str(e)}")
            datasets[split_name] = None
    
    # Create data loaders
    loaders = {}
    
    # Check if we need to set up distributed samplers
    if distributed and torch.distributed.is_initialized():
        print(f"Setting up distributed samplers for {task_name}")
        
        # Training loader with DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets[train_split],
            shuffle=shuffle_train
        )
        
        loaders['train'] = DataLoader(
            datasets[train_split],
            batch_size=batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        
        # Validation loader
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets[val_split],
            shuffle=False
        )
        
        loaders['val'] = DataLoader(
            datasets[val_split],
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        
        # Test loaders
        for test_split in test_splits:
            # Special case for backward compatibility
            if test_split == 'test_id':
                loader_name = 'test'
            else:
                loader_name = f'test_{test_split}' if not test_split.startswith('test_') else test_split
            
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                datasets[test_split],
                shuffle=False
            )
            
            loaders[loader_name] = DataLoader(
                datasets[test_split],
                batch_size=batch_size,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
    else:
        # Regular non-distributed data loaders
        # Training loader
        loaders['train'] = DataLoader(
            datasets[train_split],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        
        # Validation loader
        loaders['val'] = DataLoader(
            datasets[val_split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        
        # Test loaders
        for test_split in test_splits:
            # Special case for backward compatibility
            if test_split == 'test_id':
                loader_name = 'test'
            else:
                loader_name = f'test_{test_split}' if not test_split.startswith('test_') else test_split
            
            loaders[loader_name] = DataLoader(
                datasets[test_split],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn
            )
    
    # Get number of classes from the label mapper
    num_classes = label_mapper.num_classes
    
    # Return dictionary with additional distributed information if needed
    result = {
        'loaders': loaders,
        'datasets': datasets,
        'num_classes': num_classes,
        'label_mapper': label_mapper
    }
    
    # Include samplers in the result if distributed
    if distributed and torch.distributed.is_initialized():
        samplers = {
            'train': train_sampler,
            'val': val_sampler
        }
        # Add test samplers
        for test_split in test_splits:
            if test_split == 'test_id':
                loader_name = 'test'
            else:
                loader_name = f'test_{test_split}' if not test_split.startswith('test_') else test_split
            samplers[loader_name] = loaders[loader_name].sampler
        
        result['samplers'] = samplers
        result['is_distributed'] = True
    else:
        result['is_distributed'] = False
    
    return result
