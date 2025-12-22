import os
import sys
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import importlib.util

# Set up project paths (relative to this file)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)

# Add project root to sys.path if not already present
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import fMRI dataset module (use editable install: `pip install -e .` from project root)
from data_preparing.fmri_datasets_joint_subjects import fMRIDataset

# Configuration dictionary for model settings
MODEL_CONFIG = {
    'model_name': 'MedformerNoTSW',  # Options: 'ATMS', 'MetaEEG', 'NICE', 'EEGNetv4_Encoder', 'MindEyeModule'
    'mode': 'joint',  # Options: 'in_subject' or 'joint'
}

# Import model class based on configuration
base_path = os.path.join(_project_root, "Retrieval")

def import_from_path(module_name, file_path):
    """Helper function to import a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import the appropriate model class based on configuration
if MODEL_CONFIG['model_name'] == 'ATMS':
    atms_module = import_from_path(
        "ATMS_retrieval_joint_and_in_train_fMRI",
        f"{base_path}/ATMS_retrieval_joint_and_in_train_fMRI.py"
    )
    ModelClass = atms_module.ATMS
else:
    # For all other models, import from contrast_retrieval_fMRI.py
    contrast_module = import_from_path(
        "contrast_retrieval_fMRI",
        f"{base_path}/contrast_retrieval_fMRI.py"
    )
    
    # Map model names to their corresponding classes
    model_class_mapping = {
        'MetaEEG': contrast_module.MetaEEG,
        'NICE': contrast_module.NICE,
        'EEGNetv4_Encoder': contrast_module.EEGNetv4_Encoder,
        'MindEyeModule': contrast_module.MindEyeModule,
        'MB2CW': contrast_module.MB2CW,
        'CogcapW': contrast_module.CogcapW,
        'MindBridgeW': contrast_module.MindBridgeW,
        'NeV2L': contrast_module.NeV2L,
        'WaveW': contrast_module.WaveW,
        'MedformerNoTSW': contrast_module.MedformerNoTSW
    }
    
    ModelClass = model_class_mapping.get(MODEL_CONFIG['model_name'])
    if ModelClass is None:
        raise ValueError(f"Unknown model type: {MODEL_CONFIG['model_name']}")

# Import loss function
from loss import ClipLoss

def extract_id_from_string(s):
    """Extract numerical ID from a string (e.g., 'sub-01' -> 1)."""
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def get_fmrifeatures(sub, fmri_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    """
    Evaluate fMRI model performance on retrieval tasks.
    
    Args:
        sub: Subject identifier
        fmri_model: The fMRI model to evaluate
        dataloader: DataLoader containing test data
        device: Device to run computations on
        text_features_all: All text features for evaluation
        img_features_all: All image features for evaluation
        k: Number of classes to consider in evaluation
        eval_modality: Evaluation modality ('fmri')
        test_classes: Total number of test classes
        
    Returns:
        Tuple containing average loss, accuracy, top5 accuracy, labels, and features tensor
    """
    fmri_model.eval()
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
        for batch_idx, (_, data, labels, text, text_features, img, img_features, _, _, subject_id) in enumerate(dataloader):
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_id = extract_id_from_string(subject_id[0])
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            neural_features = fmri_model(data)
            
            logit_scale = fmri_model.logit_scale.float()            
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

# ======================================== Configuration =============================================
test_subjects = ['sub-01', 'sub-02', 'sub-03']
device_preference = 'cuda:0'
device_type = 'gpu'
data_path = "./data/fmri_dataset/Preprocessed"  # Modify according to your dataset location
device = torch.device(device_preference if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# ======================================== Configuration =============================================

# Experiment parameters
mode = MODEL_CONFIG['mode']
test_classes = 100
eval_modality = 'fmri'

# Initialize lists for storing accuracies
test_accuracies = []
test_accuracies_top5 = []
v2_accuracies = []
v4_accuracies = []
v10_accuracies = []

# Initialize dictionary to store results per subject
subject_results = {sub: {} for sub in test_subjects}

print("\n" + "="*80)
print(f"Starting experiment with following configuration:")
print(f"Model: {MODEL_CONFIG['model_name']}")
print(f"Mode: {mode}")
print(f"Test subjects: {', '.join(test_subjects)}")
print(f"Number of test classes: {test_classes}")
print(f"Evaluation modality: {eval_modality}")
print("="*80 + "\n")

# Main evaluation loop
for sub in test_subjects:
    print(f"\nProcessing subject: {sub}")
    # Load appropriate model based on mode
    fmri_model = ModelClass()
    base_path = os.path.join(_project_root, f"models/{MODEL_CONFIG['model_name']}")
    
    if mode == 'joint':
        across_dir = os.path.join(base_path, 'across', 'fMRI')
        time_folder = os.listdir(across_dir)[0]
        model_path = os.path.join(across_dir, time_folder, '40.pth')
        print(f"Loading joint model from: {model_path}")
    else:  # in_subject mode
        subject_num = sub.split('-')[1]  # Extract subject number (e.g., "01" from "sub-01")
        subject_dir = os.path.join(base_path, 'in_subject', 'fMRI', f'sub-{subject_num}')
        time_folder = os.listdir(subject_dir)[0]
        model_path = os.path.join(subject_dir, time_folder, '40.pth')
        print(f"Loading in-subject model from: {model_path}")
    
    fmri_model.load_state_dict(torch.load(model_path, map_location=device))
    fmri_model.to(device)
    fmri_model.eval()

    # Setup dataset and dataloader
    test_dataset = fMRIDataset(data_path, adap_subject=sub, subjects=test_subjects, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    text_features_test_all = test_dataset.text_features    
    img_features_test_all = test_dataset.img_features
    
    # Run evaluations with different k values
    test_loss, test_accuracy, top5_acc, labels, fmri_features_test = get_fmrifeatures(
        sub, fmri_model, test_loader, device, text_features_test_all, img_features_test_all, 
        k=test_classes, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v2_acc, _, _, _ = get_fmrifeatures(
        sub, fmri_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=2, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v4_acc, _, _, _ = get_fmrifeatures(
        sub, fmri_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=4, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v10_acc, _, _, _ = get_fmrifeatures(
        sub, fmri_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=10, eval_modality=eval_modality, test_classes=test_classes
    )    
    
    # Store results
    test_accuracies.append(test_accuracy)
    test_accuracies_top5.append(top5_acc)
    v2_accuracies.append(v2_acc)
    v4_accuracies.append(v4_acc)
    v10_accuracies.append(v10_acc)
    
    # Store individual results
    subject_results[sub] = {
        'test_acc': test_accuracy,
        'top5_acc': top5_acc,
        'v2_acc': v2_acc,
        'v4_acc': v4_acc,
        'v10_acc': v10_acc
    }
    
    print(f"\nResults for {sub}:")
    print(f" - Test Accuracy: {test_accuracy:.4f}")
    print(f" - Top5 Accuracy: {top5_acc:.4f}")    
    print(f" - v2 Accuracy: {v2_acc:.4f}")
    print(f" - v4 Accuracy: {v4_acc:.4f}")
    print(f" - v10 Accuracy: {v10_acc:.4f}")

# Print final summary
print("\n" + "="*80)
print(f"EXPERIMENT SUMMARY")
print(f"Evaluation modality: {eval_modality}")
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