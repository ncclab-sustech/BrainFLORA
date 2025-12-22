'''
@File    :   split_fmri.py
@Time    :   2025/07/13 16:18:44
@Author  :   DongyangLi
@Version :   1.0
@Desc    :   modified from [PAPER_NAME](https://arxiv.org/abs/XXXX.XXXXX) (CONFERENCE_ABBR'YY)
'''


import os
import pickle
from os.path import join as pjoin
import numpy as np
import pandas as pd

# --------------------------- Hyperparameters --------------------------- #

# List of subjects to process
# SUBJECTS = ['01', '02', '03']
SUBJECTS = ['02', '03']
# Directory paths (modify according to your dataset location)
BETAS_CSV_DIR = pjoin('./data/THINGS_fMRI/THINGS_fMRI_Single_Trial_table', 'betas_csv')
# Output directory - MODIFY PATH ACCORDING TO YOUR SETUP
OUTPUT_BASE_DIR = './data/fmri_dataset/Preprocessed'

# ROI columns of interest
ROI_COLUMNS = [
    'V1', 'V2', 'V3', 'hV4',
    'rOFA', 'lOFA',
    'rFFA', 'lFFA',
    'rEBA', 'lEBA',
    'rPPA', 'lPPA'
]

# Number of repetitions per stimulus
REPEATS_PER_STIMULUS = 12

# --------------------------- Functions --------------------------- #

def load_data(sub):
    """Load response data, voxel metadata, and stimulus metadata for a subject."""
    data_file = pjoin(BETAS_CSV_DIR, f'sub-{sub}_ResponseData.h5')
    vox_file = pjoin(BETAS_CSV_DIR, f'sub-{sub}_VoxelMetadata.csv')
    stim_file = pjoin(BETAS_CSV_DIR, f'sub-{sub}_StimulusMetadata.csv')
    
    try:
        responses = pd.read_hdf(data_file)
        voxdata = pd.read_csv(vox_file)
        stimdata = pd.read_csv(stim_file)
        print(f"Subject {sub}: Data loaded successfully.")
        return responses, voxdata, stimdata
    except FileNotFoundError as e:
        print(f"Subject {sub}: {e}")
        return None, None, None

def apply_voxel_mask(responses, voxdata):
    """Apply ROI mask to the response data."""
    vox_pick = voxdata[ROI_COLUMNS].any(axis=1)
    responses_pick = responses.to_numpy()[vox_pick, :]
    responses_pick = responses_pick[:, 1:]  # Remove the first column
    print(f"Selected voxels: {vox_pick.sum()}")
    return responses_pick

def split_and_sort_data(responses_pick, stimdata, sub):
    """Split responses into train/test based on trial type and sort by stimulus."""
    required_columns = {'trial_type', 'stimulus'}
    if not required_columns.issubset(stimdata.columns):
        raise ValueError('stimdata must contain "trial_type" and "stimulus" columns.')
    
    masks = {
        'train': stimdata['trial_type'] == 'train',
        'test': stimdata['trial_type'] == 'test'
    }
    
    sorted_data = {}
    
    for key, mask in masks.items():
        data = responses_pick[:, mask]
        stimuli = stimdata['stimulus'][mask].values
        sorted_indices = np.argsort(stimuli)
        
        sorted_data[key] = data[:, sorted_indices].transpose()
        sorted_data[key] = sorted_data[key].reshape(-1, REPEATS_PER_STIMULUS, sorted_data[key].shape[1])
        print(f"Subject {sub}: {key} data shape after sorting and reshaping: {sorted_data[key].shape}")
    
    return sorted_data

def save_to_pickle(data, filepath):
    """Save data to a pickle file."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved data to {filepath}")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")

# --------------------------- Main Processing --------------------------- #

for sub in SUBJECTS:
    print(f"\nProcessing Subject: {sub}")
    
    # Load data
    responses, voxdata, stimdata = load_data(sub)
    if responses is None or voxdata is None or stimdata is None:
        continue  # Skip to next subject if any data is missing
    
    print('Available voxel metadata columns:', voxdata.columns.tolist())
    
    # Apply voxel mask
    responses_pick = apply_voxel_mask(responses, voxdata)
    print(f"Responses shape after masking: {responses_pick.shape}")
    
    # Split and sort data
    try:
        sorted_data = split_and_sort_data(responses_pick, stimdata, sub)
    except ValueError as e:
        print(f"Subject {sub}: {e}")
        continue
    
    # Define output directory
    output_dir = pjoin(OUTPUT_BASE_DIR, f'sub-{sub}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to pickle
    save_to_pickle(sorted_data['train'], pjoin(output_dir, 'train_responses.pkl'))
    save_to_pickle(sorted_data['test'], pjoin(output_dir, 'test_responses.pkl'))
    
    print(f"Finished processing Subject: {sub}\n{'-'*50}")
