# ☁️ Weather & Cloud Type Classification with Deep Learning ☀️🌧️⚡

This repository contains a comprehensive Python script for classifying weather conditions (Clear Sky, Cloudy, Rainy, Stormy) based on cloud imagery. It leverages state-of-the-art deep learning models (Vision Transformer, ResNet, EfficientNet) and includes steps for dataset merging, preprocessing, exploratory data analysis (EDA), model training, evaluation, explainability using GradCAM, and standalone prediction.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Overview

The primary goal of this project is to automatically classify weather conditions by analyzing images of the sky. It achieves this by:

1.  Merging two distinct cloud datasets (CCSN_v2 and Howard-Cloud-X).
2.  Preprocessing the images (resizing, normalization).
3.  Splitting the data into training, validation, and test sets with a specific structure.
4.  Performing Exploratory Data Analysis (EDA) to understand data distributions.
5.  Training multiple powerful deep learning models (ViT, ResNet50, EfficientNetB0).
6.  Evaluating models and selecting the best performer based on validation accuracy.
7.  Implementing GradCAM (Gradient-weighted Class Activation Mapping) to visualize *why* the model makes certain predictions.
8.  Providing a standalone script for quick predictions on new images.

## 🚀 Features

*   **Dataset Agnostic:** Merges data from `CCSN_v2` and `Howard-Cloud-X` datasets.
*   **Automated Preprocessing:** Handles image reading, resizing, and organization.
*   **Structured Data Splits:** Creates organized `train`, `val`, and `test` directories with `weather_category/cloud_type` subfolders.
*   **In-depth EDA:** Generates visualizations (bar charts, heatmaps) of data distributions using Pandas, Matplotlib, and Seaborn.
*   **Multi-Model Training:** Trains and compares:
    *   Vision Transformer (ViT) (`vit_base_patch16_224`)
    *   ResNet50
    *   EfficientNetB0 (`tf_efficientnet_b0_ns`)
*   **Transfer Learning:** Utilizes pretrained weights for faster convergence and better performance.
*   **Early Stopping:** Prevents overfitting by monitoring validation accuracy.
*   **Best Model Saving:** Automatically saves the model with the highest validation accuracy during training and identifies the overall best model based on test performance.
*   **Explainability (XAI):** Implements GradCAM to highlight image regions crucial for classification.
*   **Standalone Prediction:** Includes a script (`Step 5`) to easily classify a single image using the best-trained model.
*   **GPU Acceleration:** Utilizes CUDA if available for significantly faster training.
*   **Robust Error Handling:** Includes checks and fallback mechanisms.

## 📁 Project Structure
├── your_repo_name/
│ ├── cloud_classifier.py # Main Python script with all steps (rename your .txt file)
│ ├── datasets/ # Directory containing input datasets (or linked via path)
│ │ ├── CCSN_v2/ # Source Dataset 1
│ │ └── Howard-Cloud-X/ # Source Dataset 2
│ ├── datasets_processed/ # OUTPUT_DIR - Where merged & processed data goes
│ │ ├── train/
│ │ │ ├── clear_sky/
│ │ │ ├── cloudy/
│ │ │ ├── rainy/
│ │ │ └── stormy/
│ │ ├── val/
│ │ │ └── ... (similar structure)
│ │ └── test/
│ │ └── ... (similar structure)
│ ├── models/ # MODEL_SAVE_PATH - Where outputs are saved
│ │ ├── class_names.json # List of determined weather classes
│ │ ├── training_results.json # Dictionary storing accuracy history and model paths
│ │ ├── best_ViT_val.pth # Example saved model (best validation)
│ │ ├── best_ResNet50_val.pth
│ │ ├── best_EfficientNetB0_val.pth
│ │ └── best_overall_model.pth # The best model across all types based on test accuracy
│ ├── requirements.txt # Python dependencies
│ └── README.md # This file

*Note: You will need to create `requirements.txt` based on the imports in the script.*

## 📊 Datasets Used

This project combines data from:

1.  **CCSN_v2 (Chinese Cloud Shape Network v2):** Contains various cloud type images. The script maps abbreviated names (e.g., 'Ci' for Cirrus) to full names and then to weather categories.
2.  **Howard-Cloud-X:** Another dataset containing cloud images, likely with different structures or labels, merged seamlessly.

The script automatically maps the inherent cloud types (like Cumulonimbus, Cirrus, Stratus) found in these datasets to the four target weather categories: `clear_sky`, `cloudy`, `rainy`, `stormy`.

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/your_repo_name.git
    cd your_repo_name
    ```

2.  **Set up a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create `requirements.txt`. Use `pip freeze > requirements.txt` after installing necessary packages like `torch`, `torchvision`, `timm`, `numpy`, `pandas`, `opencv-python`, `matplotlib`, `seaborn`, `tqdm`, `pillow`). Make sure PyTorch is installed according to your system/CUDA setup from the [official PyTorch website](https://pytorch.org/).*

4.  **Dataset Paths:**
    *   Download or locate the `CCSN_v2` and `Howard-Cloud-X` datasets.
    *   **Crucially, update the paths** inside the Python script (`cloud_classifier.py`) for `DATASET1_PATH`, `DATASET2_PATH`, `OUTPUT_DIR`, and `MODEL_SAVE_PATH` to match your environment (Google Drive or local storage).

## ▶️ Usage

1.  **Configure Paths:** Ensure the dataset and output paths at the beginning of the script (`cloud_classifier.py`) are correct for your system.

2.  **Run the Full Pipeline:** Execute the main script. This will perform all steps sequentially: merging, EDA, training, explanation, and prediction test.
    ```bash
    python cloud_classifier.py
    ```

3.  **Running Specific Parts (Optional):** The script is divided into steps (marked with `%% Step X`). You can comment out or modify sections if you only need to run specific parts (e.g., only training or only prediction).

4.  **Standalone Prediction:** Step 5 demonstrates how to load the best model (`best_overall_model.pth`) and predict the weather category for a new image file. Modify the example image path or provide input when prompted.

## 💡 Explainability with GradCAM

Step 4 utilizes GradCAM to provide insights into the model's decision-making process. It generates heatmaps that overlay the input image, highlighting the pixels and regions that were most influential in predicting a specific weather category. This helps in understanding if the model is focusing on relevant features (like specific cloud textures or formations). Example visualizations are printed during the script's execution.

*(Consider adding a sample GradCAM output image here if you have one)*
`[Placeholder for GradCAM visualization example]`

## 🏆 Results

The script trains multiple architectures and automatically saves the best weights for each model based on *validation* accuracy (`best_{ModelName}_val.pth`). It then evaluates these best validation models on the *test* set and saves the weights of the single best performing model overall as `best_overall_model.pth`.

Detailed results, including training/validation accuracy history per epoch and final test accuracies, are stored in `models/training_results.json`. The script prints a summary table at the end of Step 3.

## 🔮 Future Work

*   **Hyperparameter Tuning:** Optimize learning rates, batch sizes, optimizer settings, augmentation strategies.
*   **More Architectures:** Experiment with newer models (e.g., Swin Transformers, ConvNeXt).
*   **Larger Datasets:** Incorporate more diverse and extensive datasets.
*   **Data Balancing:** Address potential class imbalance issues using techniques like over/undersampling or weighted loss functions.
*   **Deployment:** Create a web application (using Flask/Streamlit) or API for easier interaction.
*   **Fine-grained Classification:** Expand to classify specific cloud *types* instead of broader weather categories.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file (or the badge link) for details.

## 🙏 Acknowledgements

*   The `timm` library by Ross Wightman for easy access to PyTorch image models.
*   The creators of the CCSN_v2 and Howard-Cloud-X datasets.
*   The PyTorch team and community.

---

Happy Cloud Classifying! ☁️➡️📊
