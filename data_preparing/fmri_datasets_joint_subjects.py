import os
import pickle
import itertools
import json
import requests
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)
from diffusers.utils import load_image
import open_clip

# Set up proxy and device
proxy = 'http://10.20.37.38:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
device = "cuda:5" if torch.cuda.is_available() else "cpu"

# Load configuration (relative to project root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
cfg = OmegaConf.load(os.path.join(_project_root, "configs/config.yaml"))
cfg = OmegaConf.structured(cfg)
img_directory_training = cfg.fmridataset.img_directory_training
img_directory_test = cfg.fmridataset.img_directory_test

class CLIPEncoder(nn.Module):
    """CLIP vision encoder for image feature extraction"""
    def __init__(self, device):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.clip_size = (224, 224)
        self.device = device
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], 
                            interpolation=transforms.InterpolationMode.BICUBIC, 
                            antialias=True),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def clip_encode_image(self, x):
        """Encode image patches using CLIP vision model"""
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batchsize, 1024, 256]
        x = x.permute(0, 2, 1) 

        # Prepare class embeddings
        class_embedding = self.clip.vision_model.embeddings.class_embedding.to(x.dtype)
        class_embedding = class_embedding.repeat(x.shape[0], 1, 1)  # [batchsize, 1, 1024]
        
        x = torch.cat([class_embedding, x], dim=1)
        pos_embedding = self.clip.vision_model.embeddings.position_embedding
        
        # Add positional embeddings
        position_ids = torch.arange(0, 257).unsqueeze(0).to(self.device)
        x = x + pos_embedding(position_ids)
        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)
        
        # Select features from second last layer
        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]  # [1, 256, 1024]
        image_features = select_hidden_state[:, 1:]  # Remove class token
        
        return image_features

    def encode_image(self, x):
        """Main image encoding method"""
        x = x.to(self.device)
        x = self.clip.vision_model.embeddings.patch_embedding(x)
        image_feats = self.clip_encode_image(x)
        return image_feats


class fMRIDataset(Dataset):
    """
    Dataset for fMRI data with optional text and image features
    
    Args:
        data_path (str): Path to fMRI data
        adap_subject (str): Subject ID for adaptation
        subjects (list): List of subject IDs to include
        train (bool): Whether to load training data
        use_caption (bool): Whether to use text captions
        time_window (list): Time window for EEG data
        classes (list): Specific classes to include
        pictures (list): Specific images to include
    """
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, 
                 use_caption=False, time_window=[0, 1.0], classes=None, pictures=None):
        self.data_path = data_path
        self.train = train
        self.use_caption = use_caption        
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 720 if train else 100
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject
        self.modal = 'fmri'
        
        # Validate subjects
        assert any(sub in self.subject_list for sub in self.subjects)

        # Load data
        self.data, self.labels, self.text, self.img = self.load_data()
        
        # Calculate data lengths for each subject
        self.subject_data_lens = [data.shape[0] for data in self.data]
        self.cumulative_data_lens = [0] + list(itertools.accumulate(self.subject_data_lens))
        
        # Determine feature filename based on settings
        if self.use_caption:
            model_type = 'ViT-L-14'
            features_filename = os.path.join(_current_dir, f'fMRI_{model_type}_features_multimodal_train.pt') if self.train else os.path.join(_current_dir, f'fMRI_{model_type}_features_multimodal_test.pt')
        else:
            model_type = 'ViT-H-14'     
            features_filename = os.path.join(_current_dir, f'fMRI_{model_type}_features_train.pt') if self.train else os.path.join(_current_dir, f'fMRI_{model_type}_features_test.pt')

        # Load or compute features
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=True)
            if self.use_caption:
                self.img_features = saved_features['img_features']
                self.text_features = torch.zeros((self.img_features.shape[0], 1, 1024)).cpu()
            else:
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
        else:
            if self.use_caption:                
                self.clip_encoder = CLIPEncoder(device)
                self.img_features = self.ImageEncoder(self.img, self.use_caption)
                torch.save({
                    'img_features': self.img_features.cpu(),
                    'text_features': torch.zeros((self.img_features.shape[0], 1, 1024)).cpu()
                }, features_filename)
            else:                
                self.vlmodel, self.preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
                    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
                
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)            

    def load_data(self):
        """Load fMRI data, labels, text descriptions and images"""
        data_list = []
        label_list = []
        texts = []
        images = []

        # Determine image directory based on train/test
        directory = img_directory_training if self.train else img_directory_test

        # Get all directories and sort them
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        # Create text descriptions
        for dir in dirnames:
            texts.append(f"This picture is {dir}")

        # Get all image paths
        img_directory = img_directory_training if self.train else img_directory_test
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        images = [os.path.join(os.path.join(img_directory, folder), img) 
                 for folder in all_folders 
                 for img in os.listdir(os.path.join(img_directory, folder)) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Load fMRI data for each subject
        for subject in self.subjects:
            file_name = 'train_responses.pkl' if self.train else 'test_responses.pkl'
            
            # Skip if this is not the adaptation subject in test mode
            if not self.train and subject != self.adap_subject and self.adap_subject is not None:
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
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)  # Average
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        subject_data.append(preprocessed_eeg_data_class.unsqueeze(0))
                        subject_labels.append(labels)
                        
                    data_tensor = torch.cat(subject_data, dim=0)
                    label_tensor = torch.cat(subject_labels, dim=0)

                data_list.append(data_tensor)
                label_list.append(label_tensor)

        # Print dataset statistics
        if self.train:
            for i in range(len(self.subjects)):            
                print(f"Data list length: {len(data_list[i])}, label list length: {len(label_list[i])}, "
                      f"text length: {len(texts)}, image length: {len(images)}")
        else:            
            print(f"Data list length: {len(data_list[0])}, label list length: {len(label_list[0])}, "
                  f"text length: {len(texts)}, image length: {len(images)}")    

        return data_list, label_list, texts, images

    def Textencoder(self, text):           
        """Encode text using CLIP text encoder"""
        text_inputs = torch.cat([open_clip.tokenize(t) for t in text]).to(device)
        with torch.no_grad():
            text_features = self.vlmodel.encode_text(text_inputs)
        return text_features
        
    def ImageEncoder(self, images, use_caption=False):
        """Encode images in batches"""
        batch_size = 20
        image_features_list = []
        transform = transforms.ToTensor()
        
        # Select appropriate preprocessing and encoder
        if use_caption:
            encoder = self.clip_encoder
            preprocess_fn = lambda img: self.clip_encoder.preprocess(transform(Image.open(img)))
        else:
            encoder = self.vlmodel
            preprocess_fn = lambda img: self.preprocess_train(Image.open(img).convert("RGB"))
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_fn(img) for img in batch_images])

            with torch.no_grad():
                image_features = encoder.encode_image(image_inputs)
            image_features_list.append(image_features)
            del batch_images

        return torch.cat(image_features_list, dim=0)
    

    def __getitem__(self, index):
        """Get item by index with subject-specific handling"""
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
        subject_id = self.subjects[subject_idx]  # Get subject identifier
        
        # Pad fMRI data to fixed length
        target_length = 7000
        if x.shape[0] < target_length:
            padding_size = target_length - x.shape[0]
            x = F.pad(x, (0, padding_size), value=0)
        elif x.shape[0] > target_length:
            x = x[:target_length]

        # Calculate text and image indices
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 12 * 1

        if self.train:
            text_index = (subject_offset % index_n_sub_train) // (12 * 1)
            img_index = (subject_offset % index_n_sub_train) // (1)
        else:
            text_index = (subject_offset % index_n_sub_test) // (1)
            img_index = (subject_offset % index_n_sub_test) // (1)
        
        # Get text, image and features
        text = self.text[text_index]
        img = self.img[img_index]
        if self.use_caption:
            text_features = torch.zeros((1, 1, 1024))
        else:
            text_features = self.text_features[text_index]  

        img_features = self.img_features[img_index]

        return (self.modal, x, label, text, text_features, 
                img, img_features, index, img_index, subject_id)

    def __len__(self):
        """Total length across all subjects"""
        return sum(self.subject_data_lens)


if __name__ == "__main__":
    # Example usage (modify path according to your dataset location)
    data_path = "./data/fmri_dataset/Preprocessed"
    train_dataset = fMRIDataset(data_path, subjects=['sub-01'], train=True, use_caption=True)    
    test_dataset = fMRIDataset(data_path, subjects=['sub-01'], train=False, use_caption=True)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Test sample
    i = 80*1-1
    _, x, label, text, text_features, img, img_features, index, img_index, subject_id = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)