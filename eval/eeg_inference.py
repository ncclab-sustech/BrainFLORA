import os
import sys
import re
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# Configuration dictionary for model settings
MODEL_CONFIG = {
    'model_name': 'MedformerNoTSW',  # Options: 'ATMS', 'MetaEEG', 'NICE', 'EEGNetv4_Encoder', 'MindEyeModule'
    'mode': 'joint',  # Options: 'in_subject' or 'joint'
}

# Set up project paths (relative to this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load data configuration (modify path according to your setup)
_data_config_path = os.path.join(_project_root, 'data_preparing/data_config.json')
if os.path.exists(_data_config_path):
    with open(_data_config_path, 'r') as f:
        data_config = json.load(f)
else:
    data_config = {}  # Fallback if config doesn't exist

# Import from installed package (use `pip install -e .` from project root)
from Retrieval.contrast_retrieval import (EEGNetv4_Encoder, MetaEEG, NICE, MindEyeModule, 
                               MB2CW, Cogcap, NeV2L, WaveW, MindBridgeW, MedformerNoTSW)

# Import dataset and loss function
from data_preparing.eegdatasets import EEGDataset
from utils.losses import ClipLoss


def get_model_class(model_name):
    """Return the corresponding model class based on model name.
    
    Args:
        model_name (str): Name of the model to retrieve
        
    Returns:
        class: The corresponding model class
    """
    model_mapping = {
        'MetaEEG': MetaEEG,
        'NICE': NICE,
        'EEGNetv4_Encoder': EEGNetv4_Encoder,
        'MindEyeModule': MindEyeModule,
        'MB2CW': MB2CW,
        'Cogcap': Cogcap,
        'NeV2L': NeV2L,
        'WaveW': WaveW,
        'MindBridgeW': MindBridgeW,
        'MedformerNoTSW': MedformerNoTSW,
    }
    return model_mapping.get(model_name)


def extract_id_from_string(s):
    """Extract numerical ID from subject string (e.g., 'sub-01' -> 1).
    
    Args:
        s (str): Input string containing ID
        
    Returns:
        int: Extracted numerical ID or None if not found
    """
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None


def get_eegfeatures(sub, eeg_model, dataloader, device, text_features_all, 
                   img_features_all, k, eval_modality, test_classes):
    """Evaluate EEG model performance and extract features.
    
    Args:
        sub (str): Subject ID
        eeg_model: The EEG model to evaluate
        dataloader: DataLoader for evaluation data
        device: Torch device (cpu or cuda)
        text_features_all: All text features
        img_features_all: All image features
        k: Number of classes to evaluate against
        eval_modality: Evaluation modality ('eeg')
        test_classes: Total number of test classes
        
    Returns:
        tuple: (average_loss, accuracy, top5_acc, labels, features_tensor)
    """
    eeg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    top5_correct_count = 0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    save_features = False
    features_list = []
    features_tensor = torch.zeros(0, 0)
    
    with torch.no_grad():
        for batch_idx, (_, data, labels, text, text_features, img, img_features, 
                       index, img_index, subject_id) in enumerate(dataloader):
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_id = extract_id_from_string(subject_id[0])
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            neural_features = eeg_model(data)
            
            logit_scale = eeg_model.logit_scale.float()            
            features_list.append(neural_features)
               
            img_loss = loss_func(neural_features, img_features, logit_scale)
            loss = img_loss        
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]

                logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
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
            torch.save(features_tensor.cpu(), f"ATM_S_neural_features_{sub}_train.pt")
            
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total    
    top5_acc = top5_correct_count / total    
    return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()


# ============================= Configuration ==================================
test_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 
                'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
device_preference = 'cuda:3'
device_type = 'gpu'
data_path = "./data/THINGS_EEG/Preprocessed_data_250Hz"  # Modify according to your dataset location
device = torch.device(device_preference if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# =============================================================================

# Add mode selection
mode = MODEL_CONFIG['mode']
test_classes = 200
eval_modality = 'eeg'

# Initialize lists for storing accuracies
test_accuracies = []
test_accuracies_top5 = []
v2_accuracies = []
v4_accuracies = []
v10_accuracies = []

print("\n" + "="*80)
print(f"Starting experiment with following configuration:")
print(f"Model: {MODEL_CONFIG['model_name']}")
print(f"Mode: {mode}")
print(f"Test subjects: {', '.join(test_subjects)}")
print(f"Number of test classes: {test_classes}")
print(f"Evaluation modality: {eval_modality}")
print("="*80 + "\n")

for sub in test_subjects:
    print(f"\nProcessing subject: {sub}")
    ModelClass = get_model_class(MODEL_CONFIG['model_name'])
    eeg_model = ModelClass()
    
    base_path = os.path.join(_project_root, f"models/{MODEL_CONFIG['model_name']}")
    if mode == 'joint':
        across_dir = os.path.join(base_path, 'across', 'EEG')
        time_folder = os.listdir(across_dir)[0]
        model_path = os.path.join(across_dir, time_folder, '40.pth')
        print(f"Loading joint model from: {model_path}")
    else:  # in_subject mode
        subject_num = sub.split('-')[1]
        subject_dir = os.path.join(base_path, 'in_subject', 'EEG', f'sub-{subject_num}')
        time_folder = os.listdir(subject_dir)[0]
        model_path = os.path.join(subject_dir, time_folder, '30.pth')
        print(f"Loading in-subject model from: {model_path}")
    
    eeg_model.load_state_dict(torch.load(model_path, map_location=device))
    eeg_model.to(device)
    eeg_model.eval()

    # Setup dataset and dataloader
    test_dataset = EEGDataset(data_path, adap_subject=sub, subjects=test_subjects, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    text_features_test_all = test_dataset.text_features    
    img_features_test_all = test_dataset.img_features
    
    # Run evaluations
    test_loss, test_accuracy, top5_acc, labels, eeg_features_test = get_eegfeatures(
        sub, eeg_model, test_loader, device, text_features_test_all, img_features_test_all, 
        k=test_classes, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v2_acc, _, _, _ = get_eegfeatures(
        sub, eeg_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=2, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v4_acc, _, _, _ = get_eegfeatures(
        sub, eeg_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=4, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v10_acc, _, _, _ = get_eegfeatures(
        sub, eeg_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=10, eval_modality=eval_modality, test_classes=test_classes
    )    
    
    # Store results
    test_accuracies.append(test_accuracy)
    test_accuracies_top5.append(top5_acc)
    v2_accuracies.append(v2_acc)
    v4_accuracies.append(v4_acc)
    v10_accuracies.append(v10_acc)
    
    print(f"\nResults for {sub}:")
    print(f" - Test Accuracy: {test_accuracy:.4f}")
    print(f" - Top5 Accuracy: {top5_acc:.4f}")    
    print(f" - v2 Accuracy: {v2_acc:.4f}")
    print(f" - v4 Accuracy: {v4_acc:.4f}")
    print(f" - v10 Accuracy: {v10_acc:.4f}")

print("\n" + "="*80)
print(f"EXPERIMENT SUMMARY")
print(f"Model: {MODEL_CONFIG['model_name']}")
print(f"Mode: {mode}")
print(f"Subjects: {', '.join(test_subjects)}")
print("="*80)

print("\nOverall Performance:")
print(f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
print(f"Top5 Accuracy: {np.mean(test_accuracies_top5):.4f} ± {np.std(test_accuracies_top5):.4f}")
print(f"v2 Accuracy: {np.mean(v2_accuracies):.4f} ± {np.std(v2_accuracies):.4f}")
print(f"v4 Accuracy: {np.mean(v4_accuracies):.4f} ± {np.std(v4_accuracies):.4f}")
print(f"v10 Accuracy: {np.mean(v10_accuracies):.4f} ± {np.std(v10_accuracies):.4f}")