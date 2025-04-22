# --- Confusion Matrix Visualization ---

import json
import torch
from torchvision import datasets, transforms, models
import timm # For ViT, EffNet
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import gc

# --- Configuration ---
MODEL_SAVE_PATH = Path("./models") # Path where results.json and best model are saved
OUTPUT_DIR = Path("./cloudX_improved") # Path to the processed dataset
NUM_CLASSES = 4 # Number of weather categories
CLASS_NAMES = ['clear_sky', 'cloudy', 'rainy', 'stormy'] # Make sure order is correct
IMAGE_SIZE = (224, 224)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load Class Names (Robustly) ---
if 'CLASS_NAMES' not in locals():
    try:
        class_names_path = MODEL_SAVE_PATH / "class_names.json"
        with open(class_names_path, 'r') as f:
            CLASS_NAMES = json.load(f)
        if not isinstance(CLASS_NAMES, list): raise ValueError("Invalid format")
        NUM_CLASSES = len(CLASS_NAMES)
        print(f"Loaded CLASS_NAMES: {CLASS_NAMES}")
    except Exception as e:
        print(f"Warning: Could not load class names from {class_names_path}. Using defaults. Error: {e}")
        # Fallback if loading fails
        if 'NUM_CLASSES' not in locals(): NUM_CLASSES = 4
        CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]

# --- Find Best Model from results.json ---
best_name_found = None
best_test_acc = -1.0
results_path = MODEL_SAVE_PATH / "training_results.json"
if results_path.exists():
    try:
        with open(results_path, 'r') as f:
            loaded_results = json.load(f)
        for name, res in loaded_results.items():
            current_acc = res.get('test_acc', -1)
            if isinstance(current_acc, (int, float)) and current_acc > best_test_acc:
                best_test_acc = current_acc
                best_name_found = name
        if best_name_found:
            print(f"Found best model from results: {best_name_found} (Test Acc: {best_test_acc:.4f})")
        else:
            print("Could not determine best model from results.json. Attempting default (ViT).")
            best_name_found = 'ViT' # Fallback
    except Exception as e:
        print(f"Error reading {results_path}: {e}. Attempting default (ViT).")
        best_name_found = 'ViT' # Fallback
else:
    print(f"{results_path} not found. Attempting default model (ViT).")
    best_name_found = 'ViT' # Fallback

# --- Load Best Model Architecture ---
model = None
best_model_path = MODEL_SAVE_PATH / "best_overall_model.pth"

if best_model_path.exists() and best_name_found:
    print(f"Loading architecture for {best_name_found}...")
    try:
        if best_name_found == "ViT":
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
        elif best_name_found == "ResNet50":
            model = models.resnet50(weights=None) # Use weights=None
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        elif best_name_found == "EfficientNetB0":
            model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False, num_classes=NUM_CLASSES)
        else:
            print(f"Error: Model type '{best_name_found}' not recognized for loading architecture.")

        if model:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Loaded best model weights from {best_model_path}")
        else:
            print("Failed to instantiate model architecture.")

    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Best model file not found at {best_model_path} or model name undetermined.")
    model = None

# --- Prepare Test DataLoader ---
test_loader = None
test_dir = OUTPUT_DIR / 'test'

if model is not None and test_dir.exists():
    print(f"Preparing Test DataLoader from: {test_dir}")
    try:
        # Use the same 'test' transforms as used during training/evaluation
        if 'data_transforms' not in locals() or 'test' not in data_transforms:
             print("Warning: Test data transforms not found, defining default test transforms.")
             test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])
        else:
             test_transforms = data_transforms['test'] # Use existing test transforms
        test_dataset = datasets.ImageFolder(test_dir, test_transforms)

        # Crucial check: Ensure dataset classes match expected CLASS_NAMES order
        if hasattr(test_dataset, 'classes') and test_dataset.classes != CLASS_NAMES:
            print("!!! CRITICAL WARNING: Test dataset class order mismatch !!!")
            print(f"Expected Order: {CLASS_NAMES}")
            print(f"Dataset Order: {test_dataset.classes}")
            print("Confusion matrix labels might be incorrect.")
            # Attempt to use dataset's class order if mismatch detected and seems valid
            if len(test_dataset.classes) == NUM_CLASSES:
                 print("Using dataset's class order for labels.")
                 plot_class_names = test_dataset.classes
            else:
                 plot_class_names = CLASS_NAMES # Fallback to original if lengths mismatch
        else:
            plot_class_names = CLASS_NAMES # Use original if they match or dataset has no classes attr

        if len(test_dataset) > 0:
            # Use shuffle=False for confusion matrix!
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
            print(f"Test DataLoader created with {len(test_dataset)} images.")
        else:
            print(f"Test directory '{test_dir}' contains no images.")

    except Exception as e:
        print(f"Error creating test dataset/loader: {e}")
        test_loader = None
else:
    if model is None:
        print("Skipping confusion matrix: Model not loaded.")
    if not test_dir.exists():
        print(f"Skipping confusion matrix: Test directory not found at {test_dir}")


# --- Get Predictions and True Labels ---
y_pred = []
y_true = []

if model is not None and test_loader is not None:
    print("Generating predictions on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) # Keep labels on device for now

            outputs = model(inputs)
            _, predicted_indices = torch.max(outputs, 1)

            y_pred.extend(predicted_indices.cpu().numpy()) # Move predictions to CPU and store
            y_true.extend(labels.cpu().numpy()) # Move true labels to CPU and store

    print("Predictions generated.")

    # --- Compute and Plot Confusion Matrix ---
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=plot_class_names, yticklabels=plot_class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Figure 6: Confusion Matrix for {best_name_found} on Test Set')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()
    else:
        print("Could not generate confusion matrix: No predictions or true labels found.")

# --- Cleanup (Optional) ---
if 'model' in locals() and model is not None:
    model.cpu()
    del model
    print("Model moved to CPU and deleted.")
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")