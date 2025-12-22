import os
import pickle
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from omegaconf import OmegaConf
import open_clip


# Set up device (GPU if available)
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = 'ViT-H-14'
# Initialize CLIP model
vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
    model_type, 
    pretrained='laion2b_s32b_b79k', 
    precision='fp32', 
    device=device
)
# Load configuration (relative to project root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
cfg = OmegaConf.load(os.path.join(_project_root, "configs/config.yaml"))
cfg = OmegaConf.structured(cfg)
img_directory_training = cfg.fmridataset.img_directory_training
img_directory_test = cfg.fmridataset.img_directory_test


class fMRIDataset():
    """Dataset class for fMRI data with associated images and text descriptions.
    
    Args:
        data_path (str): Path to the fMRI data
        adap_subject (str, optional): Subject ID for adaptation
        subjects (list, optional): List of subject IDs to include
        train (bool): Whether this is training data
        time_window (list): Time window for EEG extraction
        classes (list, optional): Specific classes to include
        pictures (list, optional): Specific pictures to include
    """
    
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, 
                 time_window=[0, 1.0], classes=None, pictures=None):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 720 if train else 100
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # Subject for adaptation
        self.modal = 'fmri'
        
        # Validate that specified subjects exist
        assert any(sub in self.subject_list for sub in self.subjects)

        # Load data and preprocess
        self.data, self.labels, self.text, self.img = self.load_data()
        
        # Calculate data lengths per subject
        self.subject_data_lens = [data.shape[0] for data in self.data]
        self.cumulative_data_lens = [0] + list(itertools.accumulate(self.subject_data_lens))
        
        # Load or compute features
        if self.classes is None and self.pictures is None:
            # Features file path (modify according to your setup)
            features_filename = os.path.join(_current_dir,
                               f"ori_fMRI_ViT-H-14_features_{'train' if self.train else 'test'}.pt")
            
            if os.path.exists(features_filename):
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
            else:
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)

    def load_data(self):
        """Load fMRI data along with corresponding images and text descriptions."""
        data_list = []
        label_list = []
        texts = []
        images = []

        # Determine image directory based on train/test
        directory = img_directory_training if self.train else img_directory_test

        # Get sorted list of image directories
        dirnames = [d for d in os.listdir(directory) 
                   if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        # Create text descriptions
        for dir in dirnames:
            texts.append(f"This picture is {dir}")

        # Get all image paths
        img_directory = img_directory_training if self.train else img_directory_test
        all_folders = [d for d in os.listdir(img_directory) 
                      if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        
        images = [os.path.join(os.path.join(img_directory, folder), img) 
                 for folder in all_folders 
                 for img in os.listdir(os.path.join(img_directory, folder)) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Load fMRI data for each subject
        for subject in self.subjects:
            if self.train:
                file_name = 'train_responses.pkl'
            else:
                if subject == self.adap_subject or self.adap_subject is None:
                    file_name = 'test_responses.pkl'
                else:
                    continue
                    
            file_path = os.path.join(self.data_path, subject, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                preprocessed_eeg_data = torch.from_numpy(data).float().detach()
                preprocessed_eeg_data = preprocessed_eeg_data.view(-1, *preprocessed_eeg_data.shape[1:])

                if self.train:
                    # Process training data (multiple samples per class)
                    n_classes, samples_per_class = 720, 12
                    subject_data = []
                    subject_labels = []
                    
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        subject_data.append(preprocessed_eeg_data_class)
                        subject_labels.append(labels)
                    
                    data_tensor = torch.cat(subject_data, dim=0).view(-1, *subject_data[0].shape[2:])
                    label_tensor = torch.cat(subject_labels, dim=0)
                else:
                    # Process test data (average samples per class)
                    n_classes, samples_per_class = 100, 1
                    subject_data = []
                    subject_labels = []
                    
                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:
                            continue
                            
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        subject_data.append(preprocessed_eeg_data_class.unsqueeze(0))
                        subject_labels.append(labels)
                        
                    data_tensor = torch.cat(subject_data, dim=0)
                    label_tensor = torch.cat(subject_labels, dim=0)

                data_list.append(data_tensor)
                label_list.append(label_tensor)

        # Print data statistics
        if self.train:
            for i in range(len(self.subjects)):            
                print("data_list", data_list[i].shape[1])
                print(f"Data list length: {len(data_list[i])}, label list length: {len(label_list[i])}, "
                      f"text length: {len(texts)}, image length: {len(images)}")
        else:            
            print(f"Data list length: {len(data_list[0])}, label list length: {len(label_list[0])}, "
                  f"text length: {len(texts)}, image length: {len(images)}")    

        return data_list, label_list, texts, images


    def Textencoder(self, text):   
        """
        Encode text descriptions using CLIP text encoder
        
        Args:
            text: List of text descriptions
            
        Returns:
            Normalized text features
        """
        text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        text_features = F.normalize(text_features, dim=-1).detach()
        return text_features
        
    def ImageEncoder(self, images):
        """
        Encode images using CLIP image encoder
        
        Args:
            images: List of image paths
            
        Returns:
            Normalized image features
        """
        batch_size = 20  # Process images in batches
        image_features_list = []
      
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    
    def __getitem__(self, index):
        """Get a single data sample by index.
        
        Returns:
            tuple: (modal, fmri_data, label, text, text_features, img_path, img_features, subject_id)
        """
        # Find which subject this index belongs to
        subject_idx = None
        for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
            if index < cum_len:
                subject_idx = i
                break
        subject_offset = index - self.cumulative_data_lens[subject_idx]
        
        # Get data and label
        x = self.data[subject_idx][subject_offset]
        label = self.labels[subject_idx][subject_offset]
        subject_id = self.subjects[subject_idx]  # Subject identifier
        
        # Pad fMRI data to target length
        target_length = 11000
        if x.shape[0] < target_length:
            padding_size = target_length - x.shape[0]
            x = F.pad(x, (0, padding_size), value=0)
        elif x.shape[0] > target_length:
            x = x[:target_length]

        # Calculate indices for text and image
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 12 * 1

        if self.train:
            text_index = (subject_offset % index_n_sub_train) // (12 * 1)
            img_index = (subject_offset % index_n_sub_train) // 1
        else:
            text_index = (subject_offset % index_n_sub_test) // 1
            img_index = (subject_offset % index_n_sub_test) // 1

        # Get associated text, image and features
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]

        return self.modal, x, label, text, text_features, img, img_features, subject_id

    def __len__(self):
        """Total number of samples across all subjects."""
        return sum(self.subject_data_lens)


if __name__ == "__main__":
    # Example usage (modify path according to your dataset location)
    data_path = "./data/fmri_dataset/Preprocessed"
    train_dataset = fMRIDataset(data_path, subjects=['sub-01', 'sub-02', 'sub-03'], train=True)    
    test_dataset = fMRIDataset(data_path, subjects=['sub-01', 'sub-02', 'sub-03'], train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Test sample access
    i = 80*1-1
    x, label, text, text_features, img, img_features = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)