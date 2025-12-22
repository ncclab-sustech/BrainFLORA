'''
@File    :   split_meg_rerank.py
@Time    :   2025/07/13 16:18:34
@Author  :   DongyangLi
@Version :   1.0
@Desc    :   modified from [PAPER_NAME](https://arxiv.org/abs/XXXX.XXXXX) (CONFERENCE_ABBR'YY)
'''


import os
import pickle
import numpy as np
import mne
import pandas as pd
from collections import defaultdict

# -------------------------- Hyperparameters --------------------------
# Modify these paths according to your dataset location
CSV_CONCEPT_PATH = "./data/THINGS/Metadata/Concept-specific/image_concept_index.csv"
CSV_IMAGE_PATH = "./data/THINGS/Metadata/Image-specific/image_paths.csv"

BASE_FIF_DIR = "./data/meg_dataset/original_preprocessed/preprocessed"
SOURCE_IMAGE_DIR = "./data/THINGS/Images/"
MEG_IMAGE_DIR = "./data/THINGS_MEG/images_set"
# Output directory - MODIFY PATH ACCORDING TO YOUR SETUP
OUTPUT_DIR = "./data/THINGS_MEG/preprocessed_newsplit"

CATEGORY_LIMIT = 12
SUBJECTS = [
    ('sub-01', 'preprocessed_P1-epo.fif'),
    ('sub-02', 'preprocessed_P2-epo.fif'),
    ('sub-03', 'preprocessed_P3-epo.fif'),
    ('sub-04', 'preprocessed_P4-epo.fif'),
]
# ----------------------------------------------------------------------

def load_and_crop_epochs(fif_file):
    """
    Load epochs from a FIF file, crop them to 1 second, sort by event ID,
    and remove duplicates and specific event IDs.
    """
    epochs = mne.read_epochs(fif_file, preload=True)
    epochs.crop(tmin=0, tmax=1.0)

    # Sort epochs by event ID
    sorted_indices = np.argsort(epochs.events[:, 2])
    epochs = epochs[sorted_indices]

    # Filter out epochs with event ID 999999
    filtered_epochs = epochs[epochs.events[:, 2] != 999999]

    # Remove duplicate event IDs
    unique_events, unique_indices = np.unique(filtered_epochs.events[:, 2], return_index=True)
    unique_filtered_epochs = filtered_epochs[unique_indices]

    return unique_filtered_epochs

def save_data(data, filename):
    """
    Save data to a pickle file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def count_samples_per_category(event_ids, concept_df):
    """
    Count the number of samples per category based on event IDs.
    """
    filtered_concept_df = concept_df[concept_df['Event_ID'].isin(event_ids)]
    category_counts = filtered_concept_df['Category_Label'].value_counts().sort_index()
    return category_counts.to_dict()

def build_category_prefix_mapping(meg_image_dir):
    """
    Build a mapping from category names to their prefixed counterparts.
    """
    category_prefix_mapping = {}
    for split in ["training_images", "test_images"]:
        split_dir = os.path.join(meg_image_dir, split)
        if os.path.isdir(split_dir):
            for category_with_prefix in os.listdir(split_dir):
                try:
                    prefix, category = category_with_prefix.split("_", 1)
                    category_prefix_mapping[category] = category_with_prefix
                except ValueError:
                    continue  # Skip if the folder name does not contain an underscore
    return category_prefix_mapping

def get_full_image_event_mapping(csv_img_file_path, category_prefix_mapping):
    """
    Create a mapping from event IDs to their corresponding category prefixes and image filenames.
    """
    full_image_event_mapping = {}
    df = pd.read_csv(csv_img_file_path, header=None)
    for event_id, image_path in enumerate(df[0], start=1):
        category = image_path.split("/")[1]  # Assuming the category is the folder name
        if category in category_prefix_mapping:
            category_with_prefix = category_prefix_mapping[category]
            img = os.path.basename(image_path)
            full_image_event_mapping[event_id] = (category_with_prefix, img)
    return full_image_event_mapping

def filter_and_copy_images(epochs, image_event_mapping_train, image_event_mapping_test, category_limit=12):
    """
    Filter epochs and copy corresponding images to training and testing sets based on category limits.
    """
    image_count_test = defaultdict(int)
    image_count_train = defaultdict(int)
    test_indices = []
    train_indices = []
    incomplete_categories = []

    # Sort epochs by event ID
    sorted_indices = np.argsort(epochs.events[:, 2])
    epochs = epochs[sorted_indices]

    # Initialize output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch_idx, event in enumerate(epochs.events[:, 2]):
        if event in image_event_mapping_test:
            category, img = image_event_mapping_test[event]
            if image_count_test[category] < category_limit:
                test_indices.append(epoch_idx)
                image_count_test[category] += 1
        elif event in image_event_mapping_train:
            category, img = image_event_mapping_train[event]
            if image_count_train[category] < category_limit:
                train_indices.append(epoch_idx)
                image_count_train[category] += 1

    # Extract filtered epochs
    train_epochs = epochs[train_indices]
    test_epochs = epochs[test_indices]

    # Identify incomplete categories
    for category, count in image_count_test.items():
        if count < category_limit:
            incomplete_categories.append((category, count))
    for category, count in image_count_train.items():
        if count < category_limit:
            incomplete_categories.append((category, count))

    return train_epochs, test_epochs, incomplete_categories

def process_subject(subject_id, fif_filename, concept_df, image_event_mapping_train, image_event_mapping_test):
    """
    Process a single subject: load epochs, filter data, and save processed data.
    """
    fif_file = os.path.join(BASE_FIF_DIR, fif_filename)
    epochs = load_and_crop_epochs(fif_file)
    
    category_counts = count_samples_per_category(epochs.events[:, 2], concept_df)
    print(f"Sample counts per category for {subject_id}:")
    # for category, count in category_counts.items():
    #     print(f"Category {category}: {count} samples")
    
    train_epochs, test_epochs, incomplete_categories = filter_and_copy_images(
        epochs, image_event_mapping_train, image_event_mapping_test, CATEGORY_LIMIT
    )
    
    if incomplete_categories:
        print(f"Incomplete categories for {subject_id}: {incomplete_categories}")
    
    # Create subject-specific output directory
    subject_output_dir = os.path.join(OUTPUT_DIR, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Extract and save data
    train_data = train_epochs.get_data()
    test_data = test_epochs.get_data()
    ch_names = train_epochs.ch_names
    times = train_epochs.times
    
    save_data({'meg_data': train_data, 'ch_names': ch_names, 'times': times},
              os.path.join(subject_output_dir, "preprocessed_meg_training.pkl"))
    save_data({'meg_data': test_data, 'ch_names': ch_names, 'times': times},
              os.path.join(subject_output_dir, "preprocessed_meg_test.pkl"))
    
    print(f"Processing completed for {subject_id}!")

def main():
    # Load concept labels
    concept_df = pd.read_csv(CSV_CONCEPT_PATH, header=None, names=['Category_Label'])
    concept_df['Event_ID'] = concept_df.index + 1  # Add Event_ID column

    # Build category prefix mapping
    category_prefix_mapping = build_category_prefix_mapping(MEG_IMAGE_DIR)

    # Get full image-event mapping
    full_image_event_mapping = get_full_image_event_mapping(CSV_IMAGE_PATH, category_prefix_mapping)

    # Separate training and testing mappings
    image_event_mapping_train = {}
    image_event_mapping_test = {}
    for event_id, (category, img) in full_image_event_mapping.items():
        train_path = os.path.join(MEG_IMAGE_DIR, "training_images", category, img)
        test_path = os.path.join(MEG_IMAGE_DIR, "test_images", category, img)
        if os.path.exists(train_path):
            image_event_mapping_train[event_id] = (category, img)
        elif os.path.exists(test_path):
            image_event_mapping_test[event_id] = (category, img)

    # Process each subject
    for subject_id, fif_filename in SUBJECTS:
        print(f"Starting processing for {subject_id}...")
        process_subject(subject_id, fif_filename, concept_df, image_event_mapping_train, image_event_mapping_test)

    print("All subjects have been processed successfully!")

if __name__ == "__main__":
    main()
