import sys
import os
import random
import re
import argparse
import warnings
from IPython.display import Image, display

# Import torch and related libraries
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

# Import sklearn and other utilities
from sklearn.metrics import confusion_matrix
import wandb

# Set environment variables
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"
proxy = 'http://10.20.37.38:7890'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxy

# Set up paths (relative to this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Project uses editable install - run `pip install -e .` from project root

# Import custom modules
from model.unified_encoder_multi_tower import UnifiedEncoder
from data_preparing.eegdatasets import EEGDataset
from data_preparing.megdatasets_averaged import MEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.datasets_mixer import MetaEEGDataset, MetaMEGDataset, MetafMRIDataset, MetaDataLoader
from loss import ClipLoss
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds

# Initialize wandb and ignore warnings
wandb.init(mode="disabled")
warnings.filterwarnings("ignore")

def extract_id_from_string(s):
    """Extract numerical ID from string using regex"""
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def get_eegfeatures(unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    """
    Evaluate model performance on EEG features
    
    Args:
        unified_model: The trained unified encoder model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        text_features_all: All text features
        img_features_all: All image features  
        k: Number of classes to evaluate
        eval_modality: Modality being evaluated
        test_classes: Total number of test classes
        
    Returns:
        Tuple of (average_loss, accuracy, top5_accuracy, labels, features_tensor)
    """
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    
    # Handle different modalities
    if eval_modality=='eeg' or eval_modality=='fmri':
        img_features_all = (img_features_all[eval_modality]).to(device).float()
    elif eval_modality=='meg':
        img_features_all = (img_features_all[eval_modality][::12]).to(device).float()  
    
    # Initialize metrics
    total_loss = 0
    correct = 0
    top5_correct_count=0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    save_features = False
    features_list = []  
    features_tensor = torch.zeros(0, 0)
    
    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, _, _, sub_ids) in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Get neural features from model
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            logit_scale = unified_model.logit_scale.float()            
            features_list.append(neural_features)
               
            # Calculate loss
            img_loss = loss_func(neural_features, img_features, logit_scale)
            loss = img_loss        
            total_loss += loss.item()
            
            # Calculate accuracy
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                logits_single = logits_img

                predicted_label = selected_classes[torch.argmax(logits_single).item()]
                if predicted_label == label.item():
                    correct += 1        
                
                # Calculate top-5 accuracy if needed
                if k==test_classes:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)                                                            
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                 
                total += 1              
        
    # Calculate final metrics
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total    
    top5_acc = top5_correct_count / total    
    return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()

def get_priorfeatures(sub, unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    """
    Generate and save images using diffusion prior model
    
    Args:
        sub: Subject ID
        unified_model: The trained unified encoder model
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        text_features_all: All text features
        img_features_all: All image features
        k: Number of classes to evaluate
        eval_modality: Modality being evaluated
        test_classes: Total number of test classes
        
    Returns:
        features_tensor: Tensor containing generated features
    """
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    
    # Handle different modalities
    if eval_modality=='eeg' or eval_modality=='fmri':
        img_features_all = (img_features_all[eval_modality]).to(device).float()
    elif eval_modality=='meg':
        img_features_all = (img_features_all[eval_modality][::12]).to(device).float()  
    
    # Initialize metrics and variables
    total_loss = 0
    correct = 0
    top5_correct_count=0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    save_features = False
    features_list = []  
    features_tensor = torch.zeros(0, 0)
    count = 0
    
    # Set up image generation
    seed_value = 42
    generator = Generator4Embeds(num_inference_steps=4, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_value)
    folder = os.path.join(_current_dir, f'{eval_modality}_generated_imgs')
    os.makedirs(folder, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, _, _, sub_ids) in enumerate(dataloader):
            # Move data to device
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Get neural features from model
            neural_features = unified_model(data, subject_ids, modal=eval_modality)
            
            # Generate and save images for each sample in batch
            for i in range(neural_features.shape[0]):
                h = high_pipe.generate(c_embeds=neural_features[i].unsqueeze(0), num_inference_steps=10, guidance_scale=2.0)
                image_2 = generator.generate(h, generator=gen)  
                
                # Save generated image
                dir_path = os.path.join(folder, sub)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                file_path = os.path.join(dir_path, f'image_{count+1}.png')              
                image_2.save(file_path)  
                count+=1
                
            logit_scale = unified_model.logit_scale.float()            
            features_list.append(neural_features)         

        if save_features:
            features_tensor = torch.cat(features_list, dim=0)
            print("features_tensor", features_tensor.shape)
            torch.save(features_tensor.cpu(), f"neural_features_eval_{eval_modality}_{sub}_test.pt")
            
    return features_tensor.cpu()

# Main configuration section
if __name__ == "__main__":
    # Define Parameters
    # Modify checkpoint paths according to your setup
    encoder_paths_list = [
        'eeg=./checkpoints/eeg_encoder.pth',
        'meg=./checkpoints/meg_encoder.pth',
        'fmri=./checkpoints/fmri_encoder.pth'
    ]
    
    # Evaluation configuration
    eval_modality = 'fmri'  # Modality to evaluate on
    test_subjects = ['sub-02', 'sub-03']
    modalities = ['eeg', 'meg', 'fmri']
    test_classes = 100
    
    # Dataset paths (modify according to your dataset location)
    eeg_data_path = "./data/THINGS_EEG/Preprocessed_data_250Hz"
    meg_data_path = "./data/THINGS_MEG/preprocessed_newsplit"
    fmri_data_path = "./data/fmri_dataset/Preprocessed"
    
    # Device configuration
    device_preference = 'cuda:5'
    device_type = 'gpu'
    device = torch.device(device_preference if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Process encoder_paths into a dictionary
    encoder_paths = {}
    for path in encoder_paths_list:
        key, value = path.split('=')
        encoder_paths[key] = value
    
    # Initialize empty datasets for features
    text_features_test_all = {}
    img_features_test_all = {}
    
    # Initialize the Unified Encoder Model
    unified_model = UnifiedEncoder(encoder_paths, device, user_caption=False)
    # Modify model checkpoint path according to your setup
    unified_model.load_state_dict(torch.load(os.path.join(_project_root, "checkpoints/unified_encoder.pth")))
    unified_model.to(device)
    unified_model.eval()
    
    # Initialize diffusion prior model
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1)
    high_pipe = Pipe(diffusion_prior, device=device)
    # Modify diffusion prior checkpoint path according to your setup
    high_pipe.diffusion_prior.load_state_dict(torch.load(os.path.join(_project_root, "checkpoints/prior_diffusion.pth")))
    high_pipe.diffusion_prior.to(device)
    high_pipe.diffusion_prior.eval()
    
    # Print model parameters info
    def format_num(num):
        """Format large numbers with appropriate units"""
        for unit in ['','K','M','B','T']:
            if num < 1000:
                return f"{num:.2f}{unit}"
            num /= 1000
        return f"{num:.2f}P"

    total_params = sum(p.numel() for p in unified_model.parameters())
    trainable_params = sum(p.numel() for p in unified_model.parameters() if p.requires_grad)
    print(f"Total parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")

    if total_params > 0:
        trainable_percentage = (trainable_params / total_params) * 100
        print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
    else:
        print("Total parameters count is zero, cannot compute percentage.")
    
    # Run evaluation for each test subject
    for sub in test_subjects:
        # Prepare test dataset based on eval_modality
        if eval_modality == 'eeg':
            test_dataset = EEGDataset(eeg_data_path, subjects=[sub], train=False)
        elif eval_modality == 'meg':
            test_dataset = MEGDataset(meg_data_path, subjects=[sub], train=False)
        elif eval_modality == 'fmri':
            test_dataset = fMRIDataset(fmri_data_path, adap_subject=sub, subjects=[sub], train=False)
        
        # Collect test features
        text_features_test_all[eval_modality] = test_dataset.text_features
        img_features_test_all[eval_modality] = test_dataset.img_features

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
        # Generate and save images
        eeg_features_test = get_priorfeatures(
            sub, unified_model, test_loader, device, text_features_test_all, img_features_test_all, 
            k=test_classes, eval_modality=eval_modality, test_classes=test_classes
        )