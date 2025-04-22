# -*- coding: utf-8 -*-
"""
Standalone GradCAM analysis script.

Loads the best trained model, selects specific images from the test set
(one per category, attempting to get a mix of correct/incorrect predictions),
and generates GradCAM visualizations for them.
"""

import os
import json
from pathlib import Path
import random
import gc
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm # <--- Make sure timm is installed: pip install timm
import numpy as np
import cv2 # <--- Make sure opencv is installed: pip install opencv-python-headless
import matplotlib.pyplot as plt # <--- Make sure matplotlib is installed: pip install matplotlib
from PIL import Image

# ==============================================================================
# Configuration - !!! USER MUST SET THESE PATHS !!!
# ==============================================================================
# Adjust these paths to match your project structure
MODEL_SAVE_PATH = Path("./models")       # Directory containing best_overall_model.pth, training_results.json, class_names.json
OUTPUT_DIR = Path("./cloudX_improved") # Directory containing the processed dataset (with train/val/test splits)
# ==============================================================================

# --- Basic Setup ---
IMAGE_SIZE = (224, 224)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Class Names (Robustly) ---
CLASS_NAMES = None
NUM_CLASSES = None
try:
    class_names_path = MODEL_SAVE_PATH / "class_names.json"
    if not class_names_path.exists():
        raise FileNotFoundError(f"{class_names_path} not found.")
    with open(class_names_path, 'r') as f:
        CLASS_NAMES = json.load(f)
    if not isinstance(CLASS_NAMES, list):
        raise ValueError("class_names.json should contain a list.")
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"Loaded {NUM_CLASSES} classes: {CLASS_NAMES}")
except Exception as e:
    print(f"ERROR loading class names: {e}")
    print("Please ensure 'class_names.json' exists in MODEL_SAVE_PATH and contains a list of categories.")
    # Cannot proceed without class names
    exit()

# ==============================================================================
# GradCAM Class Definition (Using Full Backward Hook)
# ==============================================================================
class GradCAM:
    """ Gradient-weighted Class Activation Mapping (Using Full Backward Hook) """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def full_backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0 and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
            else:
                # print(f"Warning: Unexpected grad_output format: {type(grad_output)}") # Optional Debug
                self.gradients = None

        self.remove_hooks() # Clear previous handles
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(full_backward_hook)
        self.hook_handles.extend([handle_forward, handle_backward])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.gradients = None
        self.activations = None

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self._register_hooks()
        tensor_device = input_tensor.device
        final_cam = np.zeros((IMAGE_SIZE[0]//16, IMAGE_SIZE[1]//16)) # Default small map on error

        try:
            self.model.zero_grad()
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = torch.argmax(output, dim=1).item()

            target_score = output[:, class_idx]
            target_score.backward(retain_graph=False)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Hooks did not capture gradients or activations.")

            gradients_cpu = self.gradients.cpu()
            activations_cpu = self.activations.cpu()

            # print(f"Debug GradCAM - Activations shape: {activations_cpu.shape}") # Keep for debug if needed
            # print(f"Debug GradCAM - Gradients shape: {gradients_cpu.shape}")   # Keep for debug if needed

            cam = None
            if gradients_cpu.ndim == 4 and activations_cpu.ndim == 4: # CNN
                weights = torch.mean(gradients_cpu, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * activations_cpu, dim=1)
                # print("Debug GradCAM: Used 4D CNN path.")
            elif gradients_cpu.ndim == 3 and activations_cpu.ndim == 3: # ViT
                weights = torch.mean(gradients_cpu, dim=1, keepdim=True)
                cam_prod = weights * activations_cpu
                cam = torch.sum(cam_prod, dim=2)
                # print("Debug GradCAM: Used 3D ViT path.")

                num_tokens = activations_cpu.shape[1]
                grid_size_maybe = np.sqrt(num_tokens - 1)
                if num_tokens > 1 and grid_size_maybe == int(grid_size_maybe): # Check if it has CLS and is square grid
                     grid_size = int(grid_size_maybe)
                     if grid_size * grid_size == num_tokens - 1:
                          # print(f"Debug GradCAM - Reshaping ViT CAM to grid size: {grid_size}x{grid_size}")
                          cam = cam[:, 1:].reshape(cam.shape[0], grid_size, grid_size)
                     else:
                         # print(f"Debug GradCAM - Token count ({num_tokens}) doesn't match expected grid. Using flat CAM.")
                         cam = cam[:, 1:] # Use 1D CAM over spatial tokens
                else:
                    # print(f"Debug GradCAM - Cannot determine grid size or no CLS token? Using flat CAM.")
                    cam = cam[:, 1:] if num_tokens > 1 else cam # Use 1D CAM or keep as is if only 1 token

            else: # Unsupported
                 print(f"Debug GradCAM - Error: Unsupported activation/gradient dimensions: Activ={activations_cpu.ndim}D, Grad={gradients_cpu.ndim}D")

            if cam is not None:
                cam = torch.relu(cam)
                cam_img = cam[0].numpy()
                min_val, max_val = np.min(cam_img), np.max(cam_img)
                if max_val > min_val:
                    cam_img = (cam_img - min_val) / (max_val - min_val + 1e-8)
                else:
                    cam_img = np.zeros_like(cam_img)
                final_cam = cam_img

        except Exception as e_call:
            print(f"!!! Error during GradCAM.__call__: {e_call}")
            import traceback
            traceback.print_exc()
        finally:
            self.remove_hooks()
        return final_cam

# ==============================================================================
# Helper Functions
# ==============================================================================
def get_layer_from_string(model, layer_str):
    """ Accesses potentially nested layer using string path """
    module = model
    parts = layer_str.split('.')
    for part in parts:
        if '[' in part and ']' in part:
            mod_name, idx = part.split('[')
            idx = int(idx.replace(']', ''))
            module = getattr(module, mod_name)[idx]
        else:
            module = getattr(module, part)
    return module

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """ Undo normalization for displaying the image. """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = tensor.clone().cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    return np.uint8(255 * img_np)

def visualize_gradcam_with_heatmap(
    image_tensor: torch.Tensor, # The single image tensor (C, H, W)
    gradcam_instance: Optional[GradCAM],
    model_to_eval: nn.Module,
    class_names_list: List[str],
    true_label_idx: int,
    target_class_idx_vis: Optional[int] = None # Option to force CAM for a specific class
    ) -> None:
    """ Visualizes original, heatmap, and overlay. """
    if model_to_eval is None:
        print("Model not available for prediction.")
        return

    # --- Prediction ---
    pred_idx = -1
    prediction_confidence = 0.0
    predicted_class = "Prediction Failed"
    try:
        input_batch = image_tensor.unsqueeze(0).to(device) # Add batch dim
        with torch.no_grad():
            outputs = model_to_eval(input_batch)
            pred_probs = torch.softmax(outputs, dim=1)
            pred_prob_tensor, pred_idx_tensor = torch.max(pred_probs, 1)
            pred_idx = pred_idx_tensor.item()
            prediction_confidence = pred_prob_tensor.item()
            predicted_class = class_names_list[pred_idx]
    except Exception as e_pred:
        print(f"Error during prediction: {e_pred}")

    true_label_name = class_names_list[true_label_idx]
    correctness = "Correct" if true_label_idx == pred_idx else "Incorrect"

    # --- Get Original Image ---
    try:
        original_img = denormalize(image_tensor)
        show_original = True
    except Exception as e_denorm:
        print(f"Error denormalizing input image: {e_denorm}")
        original_img = None
        show_original = False

    # --- Compute GradCAM ---
    overlay, heatmap_color_rgb = None, None
    gradcam_success = False
    if gradcam_instance is not None and pred_idx != -1:
        cam_target_idx = target_class_idx_vis if target_class_idx_vis is not None else pred_idx
        print(f"  Generating GradCAM for: {class_names_list[cam_target_idx]} (Index: {cam_target_idx})...")
        try:
            cam_numpy = gradcam_instance(input_batch, class_idx=cam_target_idx)
            if cam_numpy is not None and cam_numpy.size > 0 and cam_numpy.ndim >= 2:
                target_h, target_w = original_img.shape[0:2] if show_original else IMAGE_SIZE
                heatmap = cv2.resize(cam_numpy, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                heatmap_normalized = np.uint8(255 * np.clip(heatmap, 0, 1))
                heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
                heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                if show_original and original_img is not None:
                    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color_rgb, 0.4, 0)
                gradcam_success = True
            else: print("  GradCAM returned invalid output.")
        except Exception as e:
            print(f"  Error during GradCAM generation: {e}")

    # --- Display ---
    num_plots = sum([show_original, gradcam_success, gradcam_success])
    if num_plots == 0: return
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    ax = axes.ravel() if num_plots > 1 else [axes] # Ensure ax is always iterable
    plot_idx = 0

    fig.suptitle(f"Analysis | True: {true_label_name} | Pred: {predicted_class} ({prediction_confidence:.2f}) [{correctness}]", fontsize=14)

    if show_original and original_img is not None:
        ax[plot_idx].imshow(original_img)
        ax[plot_idx].set_title("Original Image")
        ax[plot_idx].axis('off'); plot_idx += 1
    elif not show_original: # Placeholder if denormalize failed
        ax[plot_idx].text(0.5, 0.5, "Original Image Error", ha='center', va='center');
        ax[plot_idx].set_title("Original Image"); ax[plot_idx].axis('off'); plot_idx += 1

    if gradcam_success and heatmap_color_rgb is not None:
        ax[plot_idx].imshow(heatmap_color_rgb); ax[plot_idx].set_title("Grad-CAM Heatmap")
        ax[plot_idx].axis('off'); plot_idx += 1
    elif plot_idx < len(ax): # Add heatmap placeholder if CAM failed but space exists
         ax[plot_idx].text(0.5, 0.5, "GradCAM Failed", ha='center', va='center')
         ax[plot_idx].set_title("Grad-CAM Heatmap"); ax[plot_idx].axis('off'); plot_idx += 1

    if gradcam_success and overlay is not None:
        ax[plot_idx].imshow(overlay); ax[plot_idx].set_title("Heatmap Overlay")
        ax[plot_idx].axis('off'); plot_idx += 1
    elif not gradcam_success and show_original and plot_idx < len(ax): # Overlay placeholder
         ax[plot_idx].text(0.5, 0.5, "Overlay Error", ha='center', va='center')
         ax[plot_idx].set_title("Heatmap Overlay"); ax[plot_idx].axis('off'); plot_idx += 1


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.show()

def get_all_test_predictions(model_to_eval, loader, class_names_list) -> List[Dict]:
    """ Get predictions, true labels, image tensors, and paths for all test samples. """
    model_to_eval.eval()
    results = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs_dev = inputs.to(device)
            labels_np = labels.numpy() # Labels from loader are usually on CPU

            outputs = model_to_eval(inputs_dev)
            _, predicted_indices = torch.max(outputs, 1)
            predicted_indices_np = predicted_indices.cpu().numpy()

            # Get file paths if possible (loader shuffle MUST be False)
            # Assumes loader is derived from ImageFolder
            try:
                 # Calculate starting index in the dataset for this batch
                 start_idx = i * loader.batch_size
                 # Get paths for the items in the current batch
                 batch_paths = [loader.dataset.samples[start_idx + j][0] for j in range(inputs.size(0))]
            except Exception:
                 # Fallback if paths aren't accessible
                 batch_paths = [f"Unknown path - batch {i} item {j}" for j in range(inputs.size(0))]

            for j in range(inputs.size(0)): # Iterate through items in the batch
                 results.append({
                     "image_tensor": inputs[j].clone(), # Store tensor (on CPU)
                     "image_path": batch_paths[j],
                     "true_label_idx": labels_np[j],
                     "true_label_name": class_names_list[labels_np[j]],
                     "predicted_label_idx": predicted_indices_np[j],
                     "predicted_label_name": class_names_list[predicted_indices_np[j]]
                 })
    return results

# ==============================================================================
# Main Execution Logic
# ==============================================================================
def main():
    print("\n--- Standalone GradCAM Analysis ---")

    # --- 1. Find and Load Best Model ---
    best_model = None
    gradcam_instance = None
    best_name_found = None
    best_test_acc = -1.0
    results_path = MODEL_SAVE_PATH / "training_results.json"
    best_model_path = MODEL_SAVE_PATH / "best_overall_model.pth"

    # Determine best model name
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                loaded_results = json.load(f)
            for name, res in loaded_results.items():
                current_acc = res.get('test_acc', -1)
                if isinstance(current_acc, (int, float)) and current_acc > best_test_acc:
                    best_test_acc = current_acc; best_name_found = name
            print(f"Determined best model: {best_name_found} (Test Acc: {best_test_acc:.4f})" if best_name_found else "Could not determine best model from results.")
        except Exception as e: print(f"Error reading {results_path}: {e}")
    else: print(f"{results_path} not found.")

    if not best_name_found:
         best_name_found = 'ViT' # Fallback assumption
         print(f"Falling back to assuming best model is: {best_name_found}")

    # Instantiate and load model
    if best_model_path.exists():
        target_layer_name_str = None
        target_layer_ref = None
        try:
            # Instantiate correct architecture
            print(f"Instantiating model architecture: {best_name_found}")
            if best_name_found == "ViT":
                best_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
                target_layer_name_str = 'blocks[-1].norm1'
            elif best_name_found == "ResNet50":
                best_model = models.resnet50(weights=None)
                best_model.fc = nn.Linear(best_model.fc.in_features, NUM_CLASSES)
                target_layer_name_str = 'layer4[-1]'
            elif best_name_found == "EfficientNetB0":
                best_model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False, num_classes=NUM_CLASSES)
                target_layer_name_str = 'conv_head' # Or check blocks[-1][-1].conv etc.
            else: raise ValueError(f"Unsupported model type: {best_name_found}")

            # Access target layer
            if target_layer_name_str:
                target_layer_ref = get_layer_from_string(best_model, target_layer_name_str)
                print(f"Identified target layer: {target_layer_name_str} ({target_layer_ref.__class__.__name__})")
            else: print("Warning: Target layer name not set for GradCAM.")

            # Load weights and setup GradCAM
            best_model.load_state_dict(torch.load(best_model_path, map_location=device))
            best_model.to(device).eval()
            print(f"Loaded model weights from {best_model_path}")

            if target_layer_ref:
                gradcam_instance = GradCAM(model=best_model, target_layer=target_layer_ref)
                print("GradCAM initialized.")
            else: print("GradCAM not initialized (no target layer).")

        except Exception as e:
            print(f"ERROR loading model or initializing GradCAM: {e}")
            best_model = None; gradcam_instance = None
    else:
        print(f"ERROR: Best model file not found at {best_model_path}.")

    if not best_model:
        print("Cannot proceed without a loaded model.")
        return

    # --- 2. Prepare Test DataLoader ---
    test_loader = None
    test_dir = OUTPUT_DIR / 'test'
    if test_dir.exists():
        print(f"Preparing Test DataLoader from: {test_dir}")
        try:
            test_transforms = transforms.Compose([ # Define required transforms
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            test_dataset = datasets.ImageFolder(test_dir, test_transforms)
            if len(test_dataset) == 0: raise ValueError("Test dataset is empty.")
            # Ensure dataset classes match expected classes
            if hasattr(test_dataset, 'classes') and test_dataset.classes != CLASS_NAMES:
                print(f"Warning: Test dataset class order mismatch! Expected={CLASS_NAMES}, Got={test_dataset.classes}")
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2) # Must have shuffle=False!
            print(f"Test DataLoader created ({len(test_dataset)} images).")
        except Exception as e: print(f"ERROR creating Test DataLoader: {e}")
    else: print(f"ERROR: Test directory not found: {test_dir}")

    if not test_loader:
        print("Cannot proceed without Test DataLoader.")
        return

    # --- 3. Get All Predictions for Test Set ---
    print("Running inference on the test set...")
    all_predictions = get_all_test_predictions(best_model, test_loader, CLASS_NAMES)
    if not all_predictions:
        print("ERROR: Failed to get predictions from the test set.")
        return
    print(f"Generated {len(all_predictions)} predictions.")

    # --- 4. Select Images for Visualization ---
    # Goal: One image per TRUE category, aiming for ~3 correct, ~1 incorrect overall.
    # Strategy: Group by true category, separate correct/incorrect, pick one per category (preferring incorrect if available and overall incorrect count < target).

    categorized_preds: Dict[int, Dict[str, List[Dict]]] = {i: {"correct": [], "incorrect": []} for i in range(NUM_CLASSES)}
    for pred_info in all_predictions:
        true_idx = pred_info["true_label_idx"]
        pred_idx = pred_info["predicted_label_idx"]
        if true_idx in categorized_preds:
             list_key = "correct" if true_idx == pred_idx else "incorrect"
             categorized_preds[true_idx][list_key].append(pred_info)

    selected_for_viz = []
    num_incorrect_needed = 1 # Aim for at least 1 incorrect
    num_incorrect_selected = 0

    for true_cat_idx in range(NUM_CLASSES):
        selected_this_cat = None
        category_data = categorized_preds.get(true_cat_idx)
        if not category_data:
             print(f"Warning: No images found for true category '{CLASS_NAMES[true_cat_idx]}' in test set.")
             continue

        # Prioritize selecting an incorrect example if needed and available
        if num_incorrect_selected < num_incorrect_needed and category_data["incorrect"]:
             selected_this_cat = random.choice(category_data["incorrect"])
             num_incorrect_selected += 1
             print(f"Selected INCORRECT for category '{CLASS_NAMES[true_cat_idx]}'")
        # Otherwise, select a correct example if available
        elif category_data["correct"]:
             selected_this_cat = random.choice(category_data["correct"])
             print(f"Selected CORRECT for category '{CLASS_NAMES[true_cat_idx]}'")
        # If no correct examples, try incorrect again (even if quota met)
        elif category_data["incorrect"]:
             selected_this_cat = random.choice(category_data["incorrect"])
             if num_incorrect_selected < num_incorrect_needed: num_incorrect_selected +=1 # Update count if it contributes
             print(f"Selected INCORRECT (backup) for category '{CLASS_NAMES[true_cat_idx]}'")
        else:
             print(f"Warning: No images found for category '{CLASS_NAMES[true_cat_idx]}' (correct or incorrect).")

        if selected_this_cat:
            selected_for_viz.append(selected_this_cat)

    if not selected_for_viz:
         print("Could not select any images for visualization.")
         return

    print(f"\nSelected {len(selected_for_viz)} images for detailed GradCAM analysis ({num_incorrect_selected} incorrect prediction(s)).")

    # --- 5. Perform and Display GradCAM for Selected Images ---
    for viz_info in selected_for_viz:
        print("-" * 60)
        print(f"Analyzing image: {Path(viz_info['image_path']).name}") # Show only filename

        visualize_gradcam_with_heatmap(
            image_tensor=viz_info["image_tensor"], # Use the stored tensor
            gradcam_instance=gradcam_instance,    # Can be None
            model_to_eval=best_model,
            class_names_list=CLASS_NAMES,
            true_label_idx=viz_info["true_label_idx"]
        )
        # Add a small pause/wait for user to see plots if needed in interactive envs
        # input("Press Enter to continue to next image...") # Uncomment if running interactively

    # --- 6. Cleanup ---
    print("\n--- Analysis Complete. Cleaning up. ---")
    if 'best_model' in locals() and best_model is not None:
        best_model.cpu()
        del best_model
    if 'gradcam_instance' in locals() and gradcam_instance is not None:
        gradcam_instance.remove_hooks()
        del gradcam_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")


# ==============================================================================
# Run the main function
# ==============================================================================
if __name__ == "__main__":
    # --- Pre-checks ---
    if not MODEL_SAVE_PATH.exists():
        print(f"ERROR: Model save path does not exist: {MODEL_SAVE_PATH}")
        print("Please set the MODEL_SAVE_PATH variable correctly.")
        exit()
    if not OUTPUT_DIR.exists():
        print(f"ERROR: Dataset output path does not exist: {OUTPUT_DIR}")
        print("Please set the OUTPUT_DIR variable correctly.")
        exit()
    if not (OUTPUT_DIR / 'test').exists():
         print(f"ERROR: Test directory not found inside output directory: {OUTPUT_DIR / 'test'}")
         exit()

    main()