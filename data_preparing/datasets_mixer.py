# Standard library imports
import itertools
import math
import random
import pickle

# Third-party imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


# Try importing custom modules with fallback to local imports
try:
    from data_preparing.eegdatasets import EEGDataset
    from data_preparing.megdatasets_averaged import MEGDataset   
    from data_preparing.fmri_datasets_joint_subjects import fMRIDataset             
except ModuleNotFoundError:
    from eegdatasets import EEGDataset  
    from megdatasets_averaged import MEGDataset 
    from fmri_datasets_joint_subjects import fMRIDataset


class PartialStratifiedBatchSampler(Sampler):
    """
    Partial Stratified Batch Sampler that ensures each batch contains 
    repeated samples of the same class.

    Args:
        labels (list or tensor): A list of labels corresponding to the dataset.
        batch_size (int): Number of samples in each batch.
        samples_per_class (int): Number of samples for each class in a batch.

    Returns:
        Batches of indices where labels are stratified and repeated.
    """
    def __init__(self, labels, batch_size, samples_per_class):
        # Validate input
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive.")
        if batch_size % samples_per_class != 0:
            raise ValueError("batch_size must be divisible by samples_per_class.")

        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        # Map labels to their corresponding indices
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # All unique labels in the dataset
        self.labels_set = list(self.label_to_indices.keys())

        # Number of classes per batch
        self.classes_per_batch = min(len(self.labels_set), self.batch_size // self.samples_per_class)

        # Ensure classes_per_batch is valid
        if self.classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be positive. Adjust batch_size or samples_per_class.")
        if len(self.labels_set) < self.classes_per_batch:
            raise ValueError("The number of unique classes is smaller than the required classes_per_batch. Reduce batch_size or samples_per_class.")

        # Shuffle the indices within each class initially
        for cls in self.labels_set:
            random.shuffle(self.label_to_indices[cls])

    def __iter__(self):
        while True:
            batch = []
            # Randomly select classes for this batch
            selected_classes = random.sample(self.labels_set, self.classes_per_batch)

            for cls in selected_classes:
                for _ in range(self.samples_per_class):
                    # Randomly select an index from the current class
                    idx = random.choice(self.label_to_indices[cls])
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

            # Handle any remaining samples (if necessary)
            if len(batch) > 0:
                yield batch

    def __len__(self):
        # Total number of batches
        return math.ceil(len(self.labels) / self.batch_size)


class MetaEEGDataset():
    """Dataset class for EEG data with meta information."""
    
    def __init__(self, eeg_data_path, eeg_subjects, train=True, use_caption=False):
        """
        Initialize EEG dataset with metadata.
        
        Args:
            eeg_data_path (str): Path to EEG data
            eeg_subjects (list): List of subject IDs
            train (bool): Whether to use training data
            use_caption (bool): Whether to use captions
        """
        self.eeg_data_path = eeg_data_path
        self.eeg_subjects = eeg_subjects
        self.n_cls = 1654 if train else 200
        self.train = train
        eeg_data = None
        self.modal = 'eeg'
        
        # Initialize base EEG dataset
        sub_eeg_dataset = EEGDataset(eeg_data_path, subjects=eeg_subjects, train=train, use_caption=use_caption)
        
        # Store dataset attributes
        self.text_features = sub_eeg_dataset.text_features
        self.img_features = sub_eeg_dataset.img_features
        self.labels = sub_eeg_dataset.labels
        self.text = sub_eeg_dataset.text
        self.img = sub_eeg_dataset.img
        self.eeg_data = sub_eeg_dataset.data

    def __getitem__(self, index):
        """Get item by index with all associated metadata."""
        eeg_data = self.eeg_data[index]
        index_n_sub_train = self.n_cls * 10 * 4
        index_n_sub_test = self.n_cls * 1 * 80

        label = self.labels[index]
        
        # Calculate text and image indices
        if self.train:
            text_index = (index % index_n_sub_train) // (10 * 4)
            img_index = (index % index_n_sub_train) // 4
        else:
            text_index = (index % index_n_sub_test)
            img_index = (index % index_n_sub_test)
            
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]

        return (self.modal, eeg_data, label, text, text_features, 
                img, img_features, index, img_index, 'sub--1')
    
    def __len__(self):
        """Return total number of samples."""
        return self.eeg_data.shape[0]


class MetaMEGDataset():
    """Dataset class for MEG data with meta information."""
    
    def __init__(self, meg_data_path, meg_subjects, train=True, use_caption=False):
        """
        Initialize MEG dataset with metadata.
        
        Args:
            meg_data_path (str): Path to MEG data
            meg_subjects (list): List of subject IDs
            train (bool): Whether to use training data
            use_caption (bool): Whether to use captions
        """
        self.meg_data_path = meg_data_path
        self.meg_subjects = meg_subjects
        self.n_cls = 1654 if train else 200
        self.train = train
        self.modal = 'meg'
        
        # Initialize base MEG dataset
        sub_meg_dataset = MEGDataset(meg_data_path, subjects=meg_subjects, train=train, use_caption=use_caption)
        
        # Store dataset attributes
        self.text_features = sub_meg_dataset.text_features
        self.img_features = sub_meg_dataset.img_features
        self.labels = sub_meg_dataset.labels
        self.text = sub_meg_dataset.text
        self.img = sub_meg_dataset.img
        self.meg_data = sub_meg_dataset.data

    def __getitem__(self, index):
        """Get item by index with all associated metadata."""
        meg_data = self.meg_data[index]
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 1 * 12        
        
        label = self.labels[index]
        
        # Calculate text and image indices
        if self.train:
            text_index = (index % index_n_sub_train) // (12 * 1)
            img_index = (index % index_n_sub_train)
        else:
            text_index = (index % index_n_sub_test)
            img_index = (index % index_n_sub_test)
            
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        
        return (self.modal, meg_data, label, text, text_features, 
                img, img_features, index, img_index, 'sub--1')
    
    def __len__(self):
        """Return total number of samples."""
        return self.meg_data.shape[0]


class MetafMRIDataset():
    """Dataset class for fMRI data with meta information."""
    
    def __init__(self, fmri_data_path, fmri_subjects, train=True, use_caption=False):
        """
        Initialize fMRI dataset with metadata.
        
        Args:
            fmri_data_path (str): Path to fMRI data
            fmri_subjects (list): List of subject IDs
            train (bool): Whether to use training data
            use_caption (bool): Whether to use captions
        """
        self.fmri_data_path = fmri_data_path
        self.fmri_subjects = fmri_subjects
        self.n_cls = 720 if train else 100
        self.train = train        
        self.modal = 'fmri'
        
        # Initialize base fMRI dataset
        sub_fmri_dataset = fMRIDataset(fmri_data_path, subjects=fmri_subjects, train=train, use_caption=use_caption)
        
        # Store dataset attributes
        self.text_features = sub_fmri_dataset.text_features
        self.img_features = sub_fmri_dataset.img_features
        self.labels = sub_fmri_dataset.labels
        self.text = sub_fmri_dataset.text
        self.img = sub_fmri_dataset.img
        self.fmri_data = sub_fmri_dataset.data               

        # Calculate the length of data for each subject     
        self.subject_data_lens = [data.shape[0] for data in self.fmri_data]
        self.cumulative_data_lens = [0] + list(itertools.accumulate(self.subject_data_lens))

    def __getitem__(self, index):
        """Get item by index with all associated metadata."""
        # Find which subject the index belongs to
        subject_idx = None
        for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
            if index < cum_len:
                subject_idx = i
                break
        subject_offset = index - self.cumulative_data_lens[subject_idx]
        
        # Get the data and label for the specific subject and offset
        x = self.fmri_data[subject_idx][subject_offset]
        label = self.labels[subject_idx][subject_offset]
        subject_id = self.fmri_subjects[subject_idx]

        # Pad the fMRI data to 7000 if necessary
        target_length = 7000
        if x.shape[0] < target_length:
            padding_size = target_length - x.shape[0]
            x = F.pad(x, (0, padding_size), value=0)
        elif x.shape[0] > target_length:
            x = x[:target_length]

        # Calculate text and img indices
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 12 * 1

        if self.train:
            text_index = (subject_offset % index_n_sub_train) // (12 * 1)
            img_index = (subject_offset % index_n_sub_train)
        else:
            text_index = (subject_offset % index_n_sub_test)
            img_index = (subject_offset % index_n_sub_test)

        # Get text, image and features
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]

        return (self.modal, x, label, text, text_features, 
                img, img_features, index, img_index, subject_id)
        
    def __len__(self):
        """Return total number of samples across all subjects."""
        return sum(self.subject_data_lens)


class MetaDataLoader:
    """Data loader that handles multiple modalities (EEG, MEG, fMRI)."""
    
    def __init__(self, eeg_dataset=None, meg_dataset=None, fmri_dataset=None, 
                 batch_size=32, is_shuffle_batch=True, shuffle=True, 
                 drop_last=False, modalities=['eeg', 'meg', 'fmri']):
        """
        Initialize multi-modal data loader.
        
        Args:
            eeg_dataset: EEG dataset object
            meg_dataset: MEG dataset object
            fmri_dataset: fMRI dataset object
            batch_size (int): Size of each batch
            is_shuffle_batch (bool): Whether to shuffle batches
            shuffle (bool): Whether to shuffle data
            drop_last (bool): Whether to drop last incomplete batch
            modalities (list): List of modalities to include
        """
        self.is_shuffle_batch = is_shuffle_batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.modalities = modalities

        # Map modalities to their corresponding datasets
        self.datasets = {
            'eeg': eeg_dataset,
            'meg': meg_dataset,
            'fmri': fmri_dataset
        }

        # Initialize DataLoaders for the selected modalities
        self.loaders = {}
        for modality in self.modalities:
            dataset = self.datasets.get(modality)
            if dataset is not None:
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size if self.is_shuffle_batch else self.batch_size // len(self.modalities),
                    shuffle=self.shuffle,
                    drop_last=drop_last
                )
                self.loaders[modality] = loader
            else:
                raise ValueError(f"No dataset provided for modality '{modality}'")

    def __iter__(self):
        """Initialize iteration."""
        # Reset the iterators at the start of each epoch
        self.iters = {modality: iter(loader) for modality, loader in self.loaders.items()}
        if self.is_shuffle_batch:
            self.modality_list = list(self.modalities)
            self.current_modality_index = 0
        return self

    def __len__(self):
        """Return total number of samples across all modalities."""
        total_samples = 0
        for modality in self.modalities:
            dataset = self.datasets.get(modality)
            if dataset is not None:
                total_samples += len(dataset)
        return total_samples

    def __next__(self):
        """Get next batch of data."""
        if self.is_shuffle_batch:
            if not self.modality_list:
                raise StopIteration
            modality = self.modality_list[self.current_modality_index]
            try:
                batch_data = next(self.iters[modality])
                self.current_modality_index = (self.current_modality_index + 1) % len(self.modality_list)
                return batch_data
            except StopIteration:
                # Remove exhausted modality
                del self.iters[modality]
                self.modality_list.remove(modality)
                if not self.modality_list:
                    raise StopIteration
                self.current_modality_index = self.current_modality_index % len(self.modality_list)
                return self.__next__()
        else:
            try:
                batch_elements = []
                for modality in self.modalities:
                    batch = next(self.iters[modality])
                    batch_elements.extend(batch)
                return tuple(batch_elements)
            except StopIteration:
                raise StopIteration


if __name__ == '__main__':
    # Example usage - MODIFY THESE PATHS ACCORDING TO YOUR SETUP
    DATA_ROOT = "./data"
    eeg_data_path = f"{DATA_ROOT}/THINGS_EEG/Preprocessed_data_250Hz"
    meg_data_path = f"{DATA_ROOT}/THINGS_MEG/preprocessed_newsplit"
    fmri_data_path = f"{DATA_ROOT}/fmri_dataset/Preprocessed"
    eeg_subjects = ['sub-01']
    meg_subjects = ['sub-01']
    fmri_subjects = ['sub-01']

    # Initialize datasets
    eegdataset = MetaEEGDataset(eeg_data_path, eeg_subjects, train=True)
    megdataset = MetaMEGDataset(meg_data_path, meg_subjects, train=True)
    fmridataset = MetafMRIDataset(fmri_data_path, meg_subjects, train=True)

    # Create MetaDataLoader
    metadataloader = MetaDataLoader(
        eeg_dataset=eegdataset, 
        meg_dataset=megdataset, 
        fmri_dataset=fmridataset,
        batch_size=64,         
    )

    # Iterate through the data loader
    for batch_idx, batch_data in enumerate(metadataloader):
        if isinstance(batch_data, tuple):
            # Process multi-modal data
            for modality_data in batch_data:
                modal, data, labels, text, text_features, img, img_features, index, img_index, subject_id = modality_data
                print(f"Batch {batch_idx + 1} - Modality: {modal}")
                print(f" - Data shape: {data.shape}")
                print(f" - Labels: {labels}")
                print(f" - Text features shape: {text_features.shape}")
                print(f" - Image features shape: {img_features.shape}")
                print(f" - Subject ID: {subject_id}")
        else:
            # Process single modality data
            modal, data, labels, text, text_features, img, img_features, index, img_index, subject_id = batch_data
            print(f"Batch {batch_idx + 1} - Modality: {modal}")
            print(f" - Data shape: {data.shape}")
            print(f" - Labels: {labels}")
            print(f" - Text features shape: {text_features.shape}")
            print(f" - Image features shape: {img_features.shape}")
            print(f" - Subject ID: {subject_id}")
        
        # Only iterate through first 5 batches for demonstration
        if batch_idx >= 5:
            break