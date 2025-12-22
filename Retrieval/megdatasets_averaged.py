import os
import pickle
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
import open_clip


# CUDA device setup
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
img_directory_training = cfg.megdataset.img_directory_training
img_directory_test = cfg.megdataset.img_directory_test


class MEGDataset():
    """
    MEG Dataset class for handling brain imaging data with associated images and text.
    
    Args:
        data_path (str): Path to the dataset
        adap_subject (str, optional): Subject to exclude during training (adaptation)
        subjects (list, optional): List of subjects to include
        train (bool): Whether this is training data
        time_window (list): Time window to extract from MEG data [start, end] in seconds
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
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # Subject to exclude during training
        self.modal = 'meg'
        
        # Load and process data
        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        
        # Load or compute text and image features
        if self.classes is None and self.pictures is None:
            # Features file path (relative to project root)
            features_filename = os.path.join(
                _project_root, 'data_preparing/newsplit_MEG_ViT-H-14_features_train.pt'
            ) if self.train else os.path.join(
                _project_root, 'data_preparing/newsplit_MEG_ViT-H-14_features_test.pt'
            )
            
            if os.path.exists(features_filename):
                saved_features = torch.load(features_filename)
                self.text_features = saved_features['text_features']
                self.img_features = saved_features['img_features']
            else:
                self.text_features = self.Textencoder(self.text)
                self.img_features = self.ImageEncoder(self.img)
                torch.save({
                    'text_features': self.text_features.cpu(),
                    'img_features': self.img_features.cpu(),
                }, features_filename)
        else:
            self.text_features = self.Textencoder(self.text)
            self.img_features = self.ImageEncoder(self.img)
            
    def load_data(self):
        """Load MEG data, labels, text descriptions and image paths"""
        data_list = []
        label_list = []
        texts = []
        images = []
        
        # Get image directories based on train/test
        directory = img_directory_training if self.train else img_directory_test
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        # Create text descriptions from directory names
        for dir in dirnames:
            new_description = f"This picture is {dir}"
            texts.append(new_description)

        # Get all image paths
        img_directory = img_directory_training if self.train else img_directory_test
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        
        images = []  # Initialize images list
        for folder in all_folders:
            folder_path = os.path.join(img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()  
            images.extend(os.path.join(folder_path, img) for img in all_images)

        # Load MEG data for each subject
        print("self.subjects", self.subjects)
        print("adap_subject", self.adap_subject)
        for subject in self.subjects:
            if self.train:
                if subject == self.adap_subject:  # Skip excluded subject
                    continue            
                file_name = 'preprocessed_meg_training.pkl'
                file_path = os.path.join(self.data_path, subject, file_name)
                print(f"{file_path}")
                
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    preprocessed_eeg_data = torch.from_numpy(data['meg_data']).float().detach()                
                    times = torch.from_numpy(data['times']).detach()
                    ch_names = data['ch_names']

                    n_classes = 1654  # Each class contains 12 images
                    samples_per_class = 12  

                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                 
            else:
                if subject == self.adap_subject or self.adap_subject is None:                                          
                    file_name = 'preprocessed_meg_test.pkl'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    
                    with open(file_path, 'rb') as file:
                        data = pickle.load(file)
                        preprocessed_eeg_data = torch.from_numpy(data['meg_data']).float().detach()
                        times = torch.from_numpy(data['times']).detach()
                        ch_names = data['ch_names']
                        n_classes = 200
                        samples_per_class = 12

                        for i in range(n_classes):
                            if self.classes is not None and i not in self.classes:
                                continue
                            start_index = i * samples_per_class
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                            labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                            preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class, 0)
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)
                else:
                    continue

        # Combine data from all subjects
        if self.train:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            label_tensor = torch.cat(label_list, dim=0)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)  
            label_tensor = torch.cat(label_list, dim=0)[::12]
            print("data_tensor", data_tensor.shape)

        # Process labels for training
        if self.train:
            label_tensor = label_tensor.repeat_interleave(1)
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
        """Extract EEG data within specified time window"""
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]
        return extracted_data
    
    def Textencoder(self, text):   
        """Encode text descriptions using CLIP model"""
        # Preprocess and encode text
        text_inputs = torch.cat([clip.tokenize(t) for t in text]).to(device)
        
        with torch.no_grad():
            text_features = vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
        return text_features
        
    def ImageEncoder(self, images):
        """Encode images using CLIP model in batches"""
        batch_size = 20  # Set appropriate batch size
        image_features_list = []
      
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            image_inputs = torch.stack([preprocess_train(Image.open(img).convert("RGB")) 
                                      for img in batch_images]).to(device)

            with torch.no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
                batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)

            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    
    def __getitem__(self, index):
        """Get item at index, returning all relevant data"""
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 12 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 12 * 1
                
            # Calculate indices for text and images
            if self.train:
                text_index = (index % index_n_sub_train) // (12 * 1)
            else:
                text_index = (index % index_n_sub_test) // (1)
                
            if self.train:
                img_index = (index % index_n_sub_train) // (1)
            else:
                img_index = (index % index_n_sub_test) // (1)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 1 * 1
                
            if self.train:
                text_index = (index % index_n_sub_train) // (1)
            else:
                text_index = (index % index_n_sub_test) // (1)
                
            if self.train:
                img_index = (index % index_n_sub_train) // (1)
            else:
                img_index = (index % index_n_sub_test) // (1)
                
        text = self.text[text_index]
        text_features = self.text_features[text_index]
        
        if self.train:
            img_features = self.img_features[img_index]
            img = self.img[img_index]
        else:
            img_features = self.img_features[::12][img_index]
            img = self.img[::12][img_index]        

        return (self.modal, x, label, text, text_features, img, img_features, 
                index, img_index, 'sub-00')

    def __len__(self):
        """Return total number of samples"""
        return self.data.shape[0]


if __name__ == "__main__":
    # Example usage
    # Example usage - MODIFY PATH ACCORDING TO YOUR SETUP
    data_path = "./data/THINGS_MEG/preprocessed_newsplit"
    train_dataset = MEGDataset(data_path, subjects=['sub-01', 'sub-02'], train=True)    
    test_dataset = MEGDataset(data_path, subjects=['sub-01'], train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Example of accessing one sample
    i = 80*1-1
    modal, x, label, text, text_features, img, img_features, _ = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)