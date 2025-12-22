import os
import pickle
import numpy as np
import mne
import pandas as pd
from collections import defaultdict
import logging
import random
import shutil

# Set logging format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_crop_epochs(fif_file):
    """Load and crop MEG event data."""
    logging.info(f"Loading MEG data: {fif_file}")
    epochs = mne.read_epochs(fif_file, preload=True)
    epochs.crop(tmin=0, tmax=1.0)
    logging.info("Cropped epochs time window to 0 to 1.0 seconds")

    # Sort epochs by event ID
    sorted_indices = np.argsort(epochs.events[:, 2])
    epochs = epochs[sorted_indices]

    # Filter out events with event ID 999999
    filtered_epochs = epochs[epochs.events[:, 2] != 999999]
    logging.info(f"Filtered out events with ID 999999, {len(filtered_epochs)} events remaining")

    # Return filtered epochs
    logging.info(f"Total {len(filtered_epochs)} epochs loaded")
    return filtered_epochs

def save_data(data, filename):
    """Save data to the specified file."""
    logging.info(f"Saving data to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def build_category_prefix_mapping(meg_image_dir):
    """Build mapping from category names to prefixed category names."""
    logging.info(f"Building category prefix mapping: {meg_image_dir}")
    category_prefix_mapping = {}
    for split in ["training_images", "test_images"]:
        split_dir = os.path.join(meg_image_dir, split)
        if os.path.isdir(split_dir):
            for category_with_prefix in os.listdir(split_dir):
                if "_" in category_with_prefix:
                    prefix, category = category_with_prefix.split("_", 1)
                    category_prefix_mapping[category] = category_with_prefix
    logging.info(f"Found {len(category_prefix_mapping)} category mappings")
    return category_prefix_mapping

def get_full_image_event_mapping(csv_img_file_path, category_prefix_mapping):
    """Create complete image event mapping based on CSV file."""
    logging.info(f"Reading image path file: {csv_img_file_path}")
    full_image_event_mapping = {}
    df = pd.read_csv(csv_img_file_path, header=None)
    for event_id, image_path in enumerate(df[0], start=1):
        parts = image_path.strip().split("/")
        if len(parts) > 1:
            category = parts[1]  # Assume category is the second part of the path
            if category in category_prefix_mapping:
                category_with_prefix = category_prefix_mapping[category]
                img = os.path.basename(image_path)
                full_image_event_mapping[event_id] = (category_with_prefix, img)
    logging.info(f"Created {len(full_image_event_mapping)} event mappings")
    return full_image_event_mapping

def exclude_event_ids_with_exact_count(epochs, count_to_exclude=12):
    """Exclude event IDs that appear exactly the specified number of times."""
    logging.info(f"Excluding event IDs that appear exactly {count_to_exclude} times")
    event_id_counts = defaultdict(int)
    for event_id in epochs.events[:, 2]:
        event_id_counts[event_id] += 1

    # Find event IDs to exclude
    event_ids_to_exclude = {event_id for event_id, count in event_id_counts.items() if count == count_to_exclude}
    logging.info(f"Total {len(event_ids_to_exclude)} event IDs will be excluded")

    # Filter epochs
    indices_to_keep = [idx for idx, event in enumerate(epochs.events) if event[2] not in event_ids_to_exclude]
    filtered_epochs = epochs[indices_to_keep]
    logging.info(f"After filtering, {len(filtered_epochs)} epochs remaining")
    return filtered_epochs

def split_categories_randomly(category_list, test_category_count, seed=42):
    """Randomly split categories into test and training sets."""
    logging.info("Starting random split of categories into test and training sets")
    random.seed(seed)
    test_categories = random.sample(category_list, test_category_count)
    train_categories = [category for category in category_list if category not in test_categories]
    logging.info(f"Test categories: {len(test_categories)}, Train categories: {len(train_categories)}")
    return test_categories, train_categories

def assign_epochs_to_sets(epochs, full_image_event_mapping, test_categories, train_categories, max_epochs_per_class=12):
    """Assign epochs to test and training sets based on categories, limiting max epochs per class."""
    logging.info("Assigning epochs to test and training sets, limiting max epochs per class")
    test_indices = []
    train_indices = []
    event_category_mapping = {}
    category_epoch_counts = defaultdict(int)

    for idx in range(len(epochs)):
        event_id = epochs.events[idx, 2]
        if event_id in full_image_event_mapping:
            category_with_prefix, _ = full_image_event_mapping[event_id]
            # Remove the original numeric prefix, keep only category name
            if "_" in category_with_prefix:
                _, category_name = category_with_prefix.split("_", 1)
            else:
                category_name = category_with_prefix
            event_category_mapping[event_id] = category_name
            if category_epoch_counts[category_name] >= max_epochs_per_class:
                continue  # Skip if exceeds maximum count
            category_epoch_counts[category_name] += 1
            if category_name in test_categories:
                test_indices.append(idx)
            elif category_name in train_categories:
                train_indices.append(idx)
    logging.info(f"After assignment - training epochs: {len(train_indices)}, test epochs: {len(test_indices)}")
    return train_indices, test_indices, event_category_mapping

def arrange_epochs_alphabetically(epochs, indices, event_ids, event_category_mapping):
    """Arrange epochs in alphabetical order by category."""
    logging.info("Arranging epochs in alphabetical order by category")
    categories = [event_category_mapping[event_id] for event_id in event_ids]
    sorted_pairs = sorted(zip(categories, indices), key=lambda x: x[0])
    sorted_indices = [idx for _, idx in sorted_pairs]
    arranged_epochs = epochs[sorted_indices]
    arranged_categories = [event_category_mapping[epochs.events[idx, 2]] for idx in sorted_indices]
    return arranged_epochs, arranged_categories

def copy_images(arranged_categories, full_image_event_mapping, event_ids, meg_image_dir, output_image_dir, set_type):
    """Copy corresponding stimulus images to specified directory."""
    logging.info(f"Starting to copy {set_type} stimulus images")
    # Create mapping from category to new numeric prefix
    unique_categories = sorted(set(arranged_categories))
    category_to_index = {category: f"{idx+1:05d}_{category}" for idx, category in enumerate(unique_categories)}

    for idx, event_id in enumerate(event_ids):
        if event_id in full_image_event_mapping:
            original_category_with_prefix, img = full_image_event_mapping[event_id]
            # Remove the original numeric prefix, keep only category name
            if "_" in original_category_with_prefix:
                _, category_name = original_category_with_prefix.split("_", 1)
            else:
                category_name = original_category_with_prefix
            if category_name in category_to_index:
                folder_name = category_to_index[category_name]
                # Try to find the image from the original training or test set directory
                src_image_path_train = os.path.join(meg_image_dir, "training_images", original_category_with_prefix, img)
                src_image_path_test = os.path.join(meg_image_dir, "test_images", original_category_with_prefix, img)
                if os.path.exists(src_image_path_train):
                    src_image_path = src_image_path_train
                elif os.path.exists(src_image_path_test):
                    src_image_path = src_image_path_test
                else:
                    logging.warning(f"Cannot find image {img}, skipping")
                    continue  # Skip if image doesn't exist

                dest_folder = os.path.join(output_image_dir, set_type, folder_name)
                os.makedirs(dest_folder, exist_ok=True)
                dest_image_path = os.path.join(dest_folder, img)
                if not os.path.exists(dest_image_path):
                    shutil.copyfile(src_image_path, dest_image_path)
    logging.info(f"{set_type} stimulus images copying completed")

def main():
    # Set paths (modify according to your dataset location)
    csv_img_file_path = "./data/THINGS/Metadata/Image-specific/image_paths.csv"
    meg_image_dir = "./data/THINGS_MEG/images_set"
    output_dir = "./data/THINGS_MEG/preprocessed_random/sub-02"
    output_image_dir = "./data/THINGS_MEG/random_image_set"
    base_fif_dir = "./data/meg_dataset/original_preprocessed/preprocessed"

    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, "test_images"), exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, "training_images"), exist_ok=True)

    # Build mappings
    category_prefix_mapping = build_category_prefix_mapping(meg_image_dir)
    full_image_event_mapping = get_full_image_event_mapping(csv_img_file_path, category_prefix_mapping)

    # Define subject list
    subjects = [
        ('sub-02', 'preprocessed_P2-epo.fif'),
        # Add more subjects here if needed
    ]

    for subject_id, fif_filename in subjects:
        logging.info(f"Starting to process subject {subject_id}")
        fif_file = os.path.join(base_fif_dir, fif_filename)
        epochs = load_and_crop_epochs(fif_file)

        # Exclude event IDs that appear exactly 12 times
        epochs = exclude_event_ids_with_exact_count(epochs, count_to_exclude=12)

        # Get all category list
        all_categories = set()
        for event_id in epochs.events[:, 2]:
            if event_id in full_image_event_mapping:
                category_with_prefix, _ = full_image_event_mapping[event_id]
                # Remove the original numeric prefix, keep only category name
                if "_" in category_with_prefix:
                    _, category_name = category_with_prefix.split("_", 1)
                else:
                    category_name = category_with_prefix
                all_categories.add(category_name)
        all_categories = sorted(all_categories)

        # Randomly split categories into test and training sets
        test_categories, train_categories = split_categories_randomly(all_categories, test_category_count=200, seed=42)

        # Assign epochs to training and test sets, limiting each category to max 12 epochs
        max_epochs_per_class = 12
        train_indices, test_indices, event_category_mapping = assign_epochs_to_sets(
            epochs, full_image_event_mapping, test_categories, train_categories, max_epochs_per_class)

        # Extract event ID lists
        train_event_ids = [epochs.events[idx, 2] for idx in train_indices]
        test_event_ids = [epochs.events[idx, 2] for idx in test_indices]

        # Arrange epochs in alphabetical order by category
        train_epochs, train_categories_ordered = arrange_epochs_alphabetically(
            epochs, train_indices, train_event_ids, event_category_mapping)
        test_epochs, test_categories_ordered = arrange_epochs_alphabetically(
            epochs, test_indices, test_event_ids, event_category_mapping)

        # Extract data
        train_data = train_epochs.get_data()
        test_data = test_epochs.get_data()
        ch_names = train_epochs.ch_names
        times = train_epochs.times

        # Count categories and epochs per category in training and test sets
        unique_train_categories = sorted(set(train_categories_ordered))
        unique_test_categories = sorted(set(test_categories_ordered))

        train_category_counts = defaultdict(int)
        for category in train_categories_ordered:
            train_category_counts[category] += 1

        test_category_counts = defaultdict(int)
        for category in test_categories_ordered:
            test_category_counts[category] += 1

        logging.info(f"Test set contains {len(unique_test_categories)} categories, {len(test_epochs)} epochs")
        logging.info(f"Training set contains {len(unique_train_categories)} categories, {len(train_epochs)} epochs")
        
        # Save data
        save_data({'meg_data': train_data, 'ch_names': ch_names, 'times': times},
                  os.path.join(output_dir, "preprocessed_meg_training.pkl"))
        save_data({'meg_data': test_data, 'ch_names': ch_names, 'times': times},
                  os.path.join(output_dir, "preprocessed_meg_test.pkl"))

        # Copy training and test set images
        copy_images(train_categories_ordered, full_image_event_mapping, train_event_ids,
                    meg_image_dir, output_image_dir, set_type="training_images")
        copy_images(test_categories_ordered, full_image_event_mapping, test_event_ids,
                    meg_image_dir, output_image_dir, set_type="test_images")

    logging.info("All subjects processing completed!")

if __name__ == "__main__":
    main()
