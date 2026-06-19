import os
import json
import numpy as np
import requests
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transformers import (
    CLIPVisionModel, 
    CLIPVisionModelWithProjection, 
    CLIPImageProcessor
)
from diffusers.utils import load_image
import open_clip

# Set up proxy and device
proxy = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load configuration (relative to project root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
cfg = OmegaConf.load(os.path.join(_project_root, "configs/config.yaml"))
cfg = OmegaConf.structured(cfg)

# Dataset paths
img_directory_training = cfg.eegdataset.img_directory_training
img_directory_test = cfg.eegdataset.img_directory_test

class CLIPEncoder(nn.Module):
    """CLIP model encoder for image features extraction"""
    
    def __init__(self, device):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.clip_size = (224, 224)
        self.device = device
        
        # Image preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], 
                            interpolation=InterpolationMode.BICUBIC, 
                            antialias=True),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def clip_encode_image(self, x):
        """Encode image patches using CLIP vision model"""
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batchsize, 1024, 256]
        x = x.permute(0, 2, 1) 

        # Add class embedding
        class_embedding = self.clip.vision_model.embeddings.class_embedding.to(x.dtype)
        class_embedding = class_embedding.repeat(x.shape[0], 1, 1)  # [batchsize, 1, 1024]
        x = torch.cat([class_embedding, x], dim=1)
        
        # Add positional embedding
        pos_embedding = self.clip.vision_model.embeddings.position_embedding
        position_ids = torch.arange(0, 257).unsqueeze(0).to(self.device)
        x = x + pos_embedding(position_ids)
        
        # Process through CLIP
        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)
        
        # Get features from second last layer
        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]  # [1, 256, 1024]
        image_features = select_hidden_state[:, 1:]  # Remove class token
        
        return image_features

    def encode_image(self, x):
        """Full image encoding pipeline"""
        x = x.to(self.device)
        x = self.preprocess(x)  # [3, 224, 224]
        x = self.clip.vision_model.embeddings.patch_embedding(x)  # [1024, 16, 16]
        image_feats = self.clip_encode_image(x)
        return image_feats


class EEGDataset():
    """
    EEG dataset loader for THINGS-EEG dataset
    Handles both training and test data with optional caption usage
    """
    
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, 
                 use_caption=False, time_window=[0, 1.0], classes=None, pictures=None):
        """
        Initialize dataset
        Args:
            data_path: Path to EEG data
            adap_subject: Subject ID for adaptation
            subjects: List of subject IDs
            train: Whether to load training data
            use_caption: Whether to use image captions
            time_window: Time window for EEG data
            classes: Specific classes to load
            pictures: Specific pictures to load
        """
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject
        self.modal = 'eeg'
        self.use_caption = use_caption
        
        # Validate subjects
        assert any(sub in self.subject_list for sub in self.subjects)
        
        # Load data
        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        
        # Define features filename based on settings
        model_type = 'ViT-L-14' if use_caption else 'ViT-H-14'
        split = 'train' if train else 'test'
        modality = 'multimodal' if use_caption else ''
        # Features file path (relative to project root)
        features_filename = os.path.join(
            _current_dir,
            f'{model_type}_features_{modality}{split}.pt'
        )
        
        # Load or compute features
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=True)
            if use_caption:
                self.img_features = saved_features['img_features']
                self.text_features = torch.zeros((self.img_features.shape[0], 1, 1024)).cpu()
            else:
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
        else:
            if use_caption:                
                self.clip_encoder = CLIPEncoder(device)
                self.img_features = self.ImageEncoder(self.img, use_caption)
                torch.save({
                    'img_features': self.img_features.cpu(),
                    'text_features': torch.zeros((self.img_features.shape[0], 1, 1024)).cpu()               
                }, features_filename)
            else:                
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)

    def load_data(self):
        """Load EEG data, labels, text descriptions and image paths"""
        data_list, label_list, texts, images = [], [], [], []
        
        # Set directory based on train/test
        directory = img_directory_training if self.train else img_directory_test
        
        # Get sorted list of class directories
        dirnames = [d for d in os.listdir(directory) 
                   if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        # Filter classes if specified
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        # Extract text descriptions from directory names
        for dir in dirnames:
            try:
                idx = dir.index('_')
                description = dir[idx+1:]
                texts.append(f"This picture is {description}")
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue
                
        # Get image paths
        img_directory = img_directory_training if self.train else img_directory_test
        all_folders = [d for d in os.listdir(img_directory) 
                      if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()

        if self.classes is not None and self.pictures is not None:
            # Specific classes and pictures
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) 
                                if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            # Specific classes only
            images = []
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) 
                                if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            # All classes
            images = []
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) 
                            if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()  
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            print("Error in image loading configuration")
            
        # Load EEG data from subjects
        print("self.subjects", self.subjects)
        print("adap_subject", self.adap_subject)
        
        for subject in self.subjects:
            if self.train:
                if subject == self.adap_subject:
                    continue            
                
                file_name = 'preprocessed_eeg_training.npy'
                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']

                n_classes = 1654
                samples_per_class = 10
                
                # Handle different data filtering cases
                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * 1 + p
                        if start_index < len(preprocessed_eeg_data):
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]
                            labels = torch.full((1,), c, dtype=torch.long).detach()
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)

            else:  # Test data
                if subject == self.adap_subject or self.adap_subject is None:
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']
                    n_classes = 200
                    samples_per_class = 1

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:
                            continue
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                else:
                    continue

        # Process loaded data into tensors
        if self.train:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            
        label_tensor = torch.cat(label_list, dim=0)
        
        # Process labels for training
        if self.train:
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                unique_values = torch.tensor(lis)        
                mapping = {val.item(): index for index, val in enumerate(unique_values)}   
                label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        self.times = times
        self.ch_names = ch_names

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, "
              f"text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):
        """Extract EEG data for specified time window"""
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]
        return extracted_data
    
    def Textencoder(self, text):   
        """Encode text descriptions using CLIP"""
        text_inputs = torch.cat([open_clip.tokenize(t) for t in text]).to(device)

        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
        return text_features

    def ImageEncoder(self, images, use_caption=False):
        """Encode images using CLIP or custom encoder"""
        batch_size = 512   
        image_features_list = []
        transform = transforms.ToTensor()
        
        if use_caption:         
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                image_inputs = torch.stack([transform(Image.open(img)) for img in batch_images])                
                
                with torch.no_grad():
                    image_feature = self.clip_encoder.encode_image(image_inputs)
                image_features_list.append(image_feature)
        else:
            vlmodel, preprocess_train, feature_extractor = open_clip.create_model_and_transforms(
                model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device=device)
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images])

                with torch.no_grad():
                    batch_image_features = vlmodel.encode_image(image_inputs)
                image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)                        
        return image_features

    def __getitem__(self, index):
        """Get item by index"""
        x = self.data[index]
        label = self.labels[index]
        
        # Calculate indices for text and image based on dataset configuration
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 10 * 4
                
            if self.train:
                text_index = (index % index_n_sub_train) // (10 * 4)
                img_index = (index % index_n_sub_train) // (4)
            else:
                text_index = (index % index_n_sub_test) 
                img_index = (index % index_n_sub_test) 
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
                
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
                img_index = (index % index_n_sub_train) // (4)
            else:
                text_index = (index % index_n_sub_test) 
                img_index = (index % index_n_sub_test) 
                
        text = self.text[text_index]
        img = self.img[img_index]
        
        if self.use_caption:
            text_features = torch.zeros((1, 1, 1024))
        else:
            text_features = self.text_features[text_index]
            
        img_features = self.img_features[img_index]
        
        return (self.modal, x, label, text, text_features, 
                img, img_features, index, img_index, 'sub-00')

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    # Example usage (modify path according to your dataset location)
    data_path = "./data/THINGS_EEG/Preprocessed_data_250Hz"
    
    # Create datasets
    train_dataset = EEGDataset(data_path, subjects=['sub-01'], train=True, use_caption=True)    
    test_dataset = EEGDataset(data_path, subjects=['sub-01'], train=False, use_caption=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Test sample
    i = 80*1-1
    _, x, label, text, text_features, img, img_features, index, img_index, _ = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
