import argparse
import pickle as pkl
import h5py
import os
import glob
import numpy as np
import re
from tqdm import tqdm


def unpickle(file):
    """
    Load the raw pickle file.
    """
    with open(file, 'rb') as fo:
        data_dict = pkl.load(fo, encoding='bytes')
    return data_dict


def load_batch(file_path, num_cases_per_batch):
    """
    Load a CIFAR-10 batch file and reshape the data.
    Returns:
        data: (num_cases_per_batch, 3, 32, 32) numpy array
        labels: (num_cases_per_batch,) numpy array
    """
    d = unpickle(file_path)
    data = d[b'data']
    labels = d[b'labels']

    # Reshape using metadata
    data = data.reshape(num_cases_per_batch, 3, 32, 32)
    
    return data, np.array(labels, dtype=np.uint8)


def natural_sort_key(path):
    """
    Key for natural sorting (e.g., data_batch_2 comes before data_batch_10).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', os.path.basename(path))]


def main():
    parser = argparse.ArgumentParser(description="Preprocess CIFAR-10 pickle files into HDF5.")
    parser.add_argument("input_dir", help="Path to the directory containing CIFAR-10 pickle files")
    parser.add_argument("--output", "-o", required=True, help="Path to the output HDF5 file")

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory.")
        return

    # Check for metadata
    meta_path = os.path.join(args.input_dir, "batches.meta")
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at {meta_path}")
        return

    meta_data = unpickle(meta_path)
    num_cases = meta_data[b'num_cases_per_batch']
    label_names = meta_data[b'label_names']
    
    # Identify batches with natural sorting
    train_batch_paths = sorted(glob.glob(os.path.join(args.input_dir, "data_batch_*")), key=natural_sort_key)
    test_batch_paths = sorted(glob.glob(os.path.join(args.input_dir, "test_batch*")), key=natural_sort_key)

    num_train = len(train_batch_paths) * num_cases
    num_test = len(test_batch_paths) * num_cases

    print(f"Found {len(train_batch_paths)} training batches and {len(test_batch_paths)} test batches.")
    print(f"Creating HDF5 file: {args.output}")

    with h5py.File(args.output, 'w') as f:
        # Store label names
        f.create_dataset('label_names', data=label_names)

        # Create groups
        train_group = f.create_group('train')
        test_group = f.create_group('test')

        # Pre-allocate datasets
        train_images = train_group.create_dataset('images', (num_train, 3, 32, 32), dtype='uint8')
        train_labels = train_group.create_dataset('labels', (num_train,), dtype='uint8')
        
        test_images = test_group.create_dataset('images', (num_test, 3, 32, 32), dtype='uint8')
        test_labels = test_group.create_dataset('labels', (num_test,), dtype='uint8')

        # Process training batches
        print("Processing training batches...")
        for i, batch_path in enumerate(tqdm(train_batch_paths)):
            data, labels = load_batch(batch_path, num_cases)
            start_idx = i * num_cases
            end_idx = start_idx + num_cases
            train_images[start_idx:end_idx] = data
            train_labels[start_idx:end_idx] = labels

        # Process test batches
        print("Processing test batches...")
        for i, batch_path in enumerate(tqdm(test_batch_paths)):
            data, labels = load_batch(batch_path, num_cases)
            start_idx = i * num_cases
            end_idx = start_idx + num_cases
            test_images[start_idx:end_idx] = data
            test_labels[start_idx:end_idx] = labels

    print("Success! Data saved to HDF5.")


if __name__ == "__main__":
    main()
