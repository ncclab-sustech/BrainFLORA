import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
from transformers import CLIPVisionModel
from diffusers.utils import load_image
import open_clip
import json
from omegaconf import OmegaConf

# Set device configuration
cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:2" if torch.cuda.is_available() else "cpu"

# Load configuration (relative to project root)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
cfg = OmegaConf.load(os.path.join(_project_root, "configs/config.yaml"))
cfg = OmegaConf.structured(cfg)
img_directory_training = cfg.megdataset.img_directory_training
img_directory_test = cfg.megdataset.img_directory_test


class CLIPEncoder(nn.Module):
    """CLIP Vision Model encoder for image feature extraction"""
    
    def __init__(self, device):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        self.clip_size = (224, 224)
        self.device = device
        
        # Image preprocessing pipeline
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], 
                            interpolation=transforms.InterpolationMode.BICUBIC, 
                            antialias=True),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        
    def clip_encode_image(self, x):
        """Encode image patches using CLIP model"""
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
        
        # Select features from second-to-last layer
        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]  # [1, 256, 1024]
        
        image_features = select_hidden_state[:, 1:]  # [1, 256, 1024]
        return image_features

    def encode_image(self, x):
        """Main image encoding method"""
        x = x.to(self.device)
        x = self.clip.vision_model.embeddings.patch_embedding(x)  # [1024, 16, 16]
        image_feats = self.clip_encode_image(x)
        return image_feats


class MEGDataset():
    """
    Dataset class for MEG (Magnetoencephalography) data
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
    """
    
    def __init__(self, data_path, adap_subject=None, subjects=None, train=True, 
                use_caption=False, time_window=[0, 1.0], classes=None, pictures=None):
        """
        Initialize MEG dataset
        
        Args:
            data_path: Path to MEG data
            adap_subject: Subject to exclude during training (for adaptation)
            subjects: List of subjects to include
            train: Whether this is training data
            use_caption: Whether to use text captions
            time_window: Time window to extract from MEG signals
            classes: Specific classes to include
            pictures: Specific pictures to include
        """
        self.data_path = data_path
        self.train = train
        self.use_caption = use_caption
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject
        self.modal = 'meg'
        
        # Load and process data
        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        
        # Define feature file paths based on configuration
        if self.use_caption:
            model_type = 'ViT-L-14'
            # Features file path (relative to this directory)
            features_filename = os.path.join(
                _current_dir, f'newsplit_MEG_{model_type}_features_multimodal_train.pt' 
                if self.train else 
                f'newsplit_MEG_{model_type}_features_multimodal_test.pt')
        else:
            model_type = 'ViT-H-14'     
            # Features file path (relative to this directory)
            features_filename = os.path.join(
                _current_dir, f'newsplit_MEG_{model_type}_features_train.pt' 
                if self.train else 
                f'newsplit_MEG_{model_type}_features_test.pt')

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
        """Load MEG data, labels, text descriptions and image paths"""
        data_list = []
        label_list = []
        texts = []
        images = []
        
        # Get image directory based on train/test mode
        directory = img_directory_training if self.train else img_directory_test
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        # Create text descriptions
        for dir in dirnames:
            new_description = f"This picture is {dir}"
            texts.append(new_description)

        # Get all image paths
        img_directory = img_directory_training if self.train else img_directory_test
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()
        
        images = []
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

                    n_classes = 1654  # Each class contains 10 images
                    samples_per_class = 12  # 12 samples per class

                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index + samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
            else:
                if subject == self.adap_subject or self.adap_subject==None:                                          
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

        # Process loaded data into tensors
        if self.train:
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            label_tensor = torch.cat(label_list, dim=0)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)  
            label_tensor = torch.cat(label_list, dim=0)[::12]
            print("data_tensor", data_tensor.shape)

        # Remap labels if specific classes were selected
        if self.train and self.classes is not None:
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

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):
        """Extract EEG data within specified time window"""
        start, end = time_window
        indices = (self.times >= start) & (self.times <= end)
        extracted_data = eeg_data[..., indices]
        return extracted_data
    
    def Textencoder(self, text):           
        """Encode text using CLIP model"""
        text_inputs = torch.cat([open_clip.tokenize(t) for t in text]).to(device)

        with torch.no_grad():
            text_features = self.vlmodel.encode_text(text_inputs)
        
        text_features = F.normalize(text_features, dim=-1).detach()
        return text_features
        
    def ImageEncoder(self, images, use_caption=False):
        """Encode images using either CLIP or vision-language model"""
        batch_size = 256
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

        # Concatenate all features
        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    
    def __getitem__(self, index):
        """Get item by index"""
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 12 * 1
                index_n_sub_test = self.n_cls * 1 * 12
            else:
                index_n_sub_test = len(self.classes)* 1 * 12
                index_n_sub_train = len(self.classes)* 12 * 1
                
            # Calculate text and image indices
            if self.train:
                text_index = (index % index_n_sub_train) // (12 * 1)
                img_index = (index % index_n_sub_train) // (1)
            else:
                text_index = (index % index_n_sub_test) // (1)
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
                img_index = (index % index_n_sub_train) // (1)
            else:
                text_index = (index % index_n_sub_test) // (1)
                img_index = (index % index_n_sub_test) // (1)
                
        text = self.text[text_index]
        
        if self.use_caption:
            text_features = torch.zeros((1, 1, 1024))
        else:
            text_features = self.text_features[text_index]        
            
        if self.train:
            img_features = self.img_features[img_index]
            img = self.img[img_index]
        else:
            img_features = self.img_features[::12][img_index]
            img = self.img[::12][img_index]        
            
        return self.modal, x, label, text, text_features, img, img_features, index, img_index, 'sub-00'

    def __len__(self):
        """Get dataset length"""
        return self.data.shape[0]


if __name__ == "__main__":
    # Example usage (modify path according to your dataset location)
    data_path = "./data/THINGS_MEG/preprocessed_newsplit"
    train_dataset = MEGDataset(data_path, subjects=['sub-01'], train=True, use_caption=True)    
    test_dataset = MEGDataset(data_path, subjects=['sub-01'], train=False, use_caption=True)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Test sample
    i = 80*1-1
    modal, x, label, text, text_features, img, img_features, index, img_index, _ = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)