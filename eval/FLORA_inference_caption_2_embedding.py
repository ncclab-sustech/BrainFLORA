import sys
import os
import random
import re
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import wandb
from sklearn.metrics import confusion_matrix

# Local imports
from model.unified_encoder_multi_tower import UnifiedEncoder
from data_preparing.eegdatasets import EEGDataset
from data_preparing.megdatasets_averaged import MEGDataset
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
from data_preparing.datasets_mixer import MetaEEGDataset, MetaMEGDataset, MetafMRIDataset, MetaDataLoader
from loss import ClipLoss
from model.diffusion_prior import Pipe, EmbeddingDataset, DiffusionPriorUNet
from model.custom_pipeline import Generator4Embeds

# Set environment variables
os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore")  # Ignore warnings
wandb.init(mode="disabled")  # Disable wandb

# Set up project paths (relative to this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Project uses editable install - run `pip install -e .` from project root

def extract_id_from_string(s):
    """Extract numerical ID from string using regex.
    
    Args:
        s (str): Input string containing an ID at the end
        
    Returns:
        int: Extracted numerical ID or None if not found
    """
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def get_eegfeatures(unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    """Extract EEG features and evaluate model performance.
    
    Args:
        unified_model: The trained unified encoder model
        dataloader: DataLoader for evaluation data
        device: Device to run computation on
        text_features_all: Precomputed text features
        img_features_all: Precomputed image features
        k: Number of classes to evaluate against
        eval_modality: Current evaluation modality ('eeg', 'meg' or 'fmri')
        test_classes: Total number of test classes
        
    Returns:
        tuple: (average_loss, accuracy, top5_acc, labels, features_tensor)
    """
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float()
    
    # Handle different modalities' image features
    if eval_modality == 'eeg' or eval_modality == 'fmri':
        img_features_all = img_features_all[eval_modality].to(device).float()
    elif eval_modality == 'meg':
        img_features_all = img_features_all[eval_modality][::12].to(device).float()
        
    # Initialize metrics
    total_loss = 0
    correct = 0
    top5_correct_count = 0
    total = 0
    loss_func = ClipLoss()
    all_labels = set(range(text_features_all.size(0)))
    save_features = True
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
            
            # Get model outputs
            ret_emb, neural_features = unified_model(data, subject_ids, modal=eval_modality)
            logit_scale = unified_model.logit_scale.float()
            features_list.append(neural_features)
            
            # Calculate loss
            img_loss = loss_func(ret_emb, img_features, logit_scale)
            loss = img_loss
            total_loss += loss.item()
            
            # Evaluate accuracy
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                logits_img = logit_scale * ret_emb[idx] @ selected_img_features.T
                logits_single = logits_img
                
                predicted_label = selected_classes[torch.argmax(logits_single).item()]
                if predicted_label == label.item():
                    correct += 1
                    
                if k == test_classes:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                total += 1
                
        if save_features:
            features_tensor = torch.cat(features_list, dim=0)
            print("features_tensor", features_tensor.shape)
            torch.save(features_tensor.cpu(), f"test.pt")
    
    # Calculate final metrics
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    
    return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()

def main():
    """Main execution function for model evaluation."""
    # Configuration parameters
    # Modify checkpoint paths according to your setup
    encoder_paths_list = [
        'eeg=./checkpoints/eeg_encoder.pth',
        'meg=./checkpoints/meg_encoder.pth',
        'fmri=./checkpoints/fmri_encoder.pth'
    ]
    
    # Evaluation configuration
    eval_modality = 'fmri'
    test_subjects = ['sub-03']
    eeg_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    meg_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
    fmri_subjects = ['sub-01', 'sub-02', 'sub-03']
    modalities = ['eeg', 'meg', 'fmri']
    test_classes = 100
    
    # Dataset paths (modify according to your dataset location)
    eeg_data_path = "./data/THINGS_EEG/Preprocessed_data_250Hz"
    meg_data_path = "./data/THINGS_MEG/preprocessed_newsplit"
    fmri_data_path = "./data/fmri_dataset/Preprocessed"
    
    # Device configuration
    device_preference = 'cuda:4'
    device_type = 'gpu'
    device = torch.device(device_preference if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Process encoder paths
    encoder_paths = {}
    for path in encoder_paths_list:
        key, value = path.split('=')
        encoder_paths[key] = value
    
    # Initialize model
    unified_model = UnifiedEncoder(encoder_paths, device, user_caption=True)
    # Modify model checkpoint path according to your setup
    unified_model.load_state_dict(torch.load(os.path.join(_project_root, "checkpoints/unified_encoder_caption.pth")))
    unified_model.to(device)
    unified_model.eval()
    
    # Print model info
    def format_num(num):
        """Format large numbers with appropriate units."""
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
    
    # Evaluation loop
    text_features_test_all = {}
    img_features_test_all = {}
    test_accuracies = []
    test_accuracies_top5 = []
    v2_accuracies = []
    v4_accuracies = []
    v10_accuracies = []
    
    for sub in test_subjects:
        # Prepare test dataset
        if eval_modality == 'eeg':
            test_dataset = EEGDataset(eeg_data_path, subjects=[sub], train=False)
        elif eval_modality == 'meg':
            test_dataset = MEGDataset(meg_data_path, subjects=[sub], train=False)
        elif eval_modality == 'fmri':
            test_dataset = fMRIDataset(fmri_data_path, adap_subject=sub, subjects=[sub], train=False)
        
        # Collect features
        text_features_test_all[eval_modality] = test_dataset.text_features
        img_features_test_all[eval_modality] = test_dataset.img_features
        
        # Create dataloader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
        # Evaluate at different k values
        test_loss, test_accuracy, top5_acc, labels, eeg_features_test = get_eegfeatures(
            unified_model, test_loader, device, text_features_test_all, img_features_test_all, 
            k=test_classes, eval_modality=eval_modality, test_classes=test_classes
        )
        _, v2_acc, _, _, _ = get_eegfeatures(
            unified_model, test_loader, device, text_features_test_all, img_features_test_all, 
            k=2, eval_modality=eval_modality, test_classes=test_classes
        )
        _, v4_acc, _, _, _ = get_eegfeatures(
            unified_model, test_loader, device, text_features_test_all, img_features_test_all, 
            k=4, eval_modality=eval_modality, test_classes=test_classes
        )
        _, v10_acc, _, _, _ = get_eegfeatures(
            unified_model, test_loader, device, text_features_test_all, img_features_test_all, 
            k=10, eval_modality=eval_modality, test_classes=test_classes
        )
        
        # Store results
        test_accuracies.append(test_accuracy)
        test_accuracies_top5.append(top5_acc)
        v2_accuracies.append(v2_acc)
        v4_accuracies.append(v4_acc)
        v10_accuracies.append(v10_acc)
        
        # Print results
        print(f" - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")    
        print(f" - Test Loss: {test_loss:.4f}, v2_acc Accuracy: {v2_acc:.4f}")
        print(f" - Test Loss: {test_loss:.4f}, v4_acc Accuracy: {v4_acc:.4f}")
        print(f" - Test Loss: {test_loss:.4f}, v10_acc Accuracy: {v10_acc:.4f}")
    
    # Calculate and print average metrics
    average_test_accuracy = np.mean(test_accuracies)
    average_test_accuracy_top5 = np.mean(test_accuracies_top5)
    average_v2_acc = np.mean(v2_accuracies)
    average_v4_acc = np.mean(v4_accuracies)
    average_v10_acc = np.mean(v10_accuracies)
    
    print(f"\nAverage Test Accuracy across all subjects: {average_test_accuracy:.4f}")
    print(f"\nAverage Test Top5 Accuracy across all subjects: {average_test_accuracy_top5:.4f}")
    print(f"Average v2_acc Accuracy across all subjects: {average_v2_acc:.4f}")
    print(f"Average v4_acc Accuracy across all subjects: {average_v4_acc:.4f}")
    print(f"Average v10_acc Accuracy across all subjects: {average_v10_acc:.4f}")
    
    # Save features
    eeg_features_test.shape
    torch.save(eeg_features_test, os.path.join(_current_dir, 'fMRI_features_sub_03_test.pt'))

if __name__ == "__main__":
    main()