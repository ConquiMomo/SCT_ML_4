# SCT_ML_4 — Hand Gesture Recognition 📸✋

## Overview

The goal of **SCT_ML_4** is to implement a hand gesture recognition model that can accurately identify and classify different hand gestures (e.g., letters A–Z, space, delete, “nothing”, etc.) from image or video data. This enables intuitive human–computer interaction and gesture-based control systems.  

This project was developed as part of the SkillCraft Technology internship program.

## Table of Contents

- [Project Structure](#project-structure)  
- [Dataset](#dataset)  
- [Dependencies & Setup](#dependencies--setup)  
- [Usage](#usage)  
- [Training the Model](#training-the-model)  
- [Testing / Inference](#testing--inference)  
- [Results](#results)  
- [Future Work](#future-work)  
- [License](#license)  

---

## Project Structure
SCT_ML_4/
│
├── data/
│ ├── train/
│ ├── val/
│ └── test/
│
├── src/
│ ├── data_preprocessing.py
│ ├── data_augmentation.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py # inference / webcam script
│ └── utils.py
│
├── notebooks/ # optional — EDA / experiments
│ ├── data_exploration.ipynb
│ └── training_history.ipynb
│
├── requirements.txt
├── README.md # ← you are here
└── LICENSE


*(Note: extend this if you add data folders, notebooks, saved models, etc.)*

## ✅ Dependencies & Setup

To run the project, you need:

- Python 3.7+ (or compatible)  
- Dependencies listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt
```
📂 Dataset
- The dataset used is the ASL Alphabet Dataset 
- Structure: images organized into class subfolders (A–Z, space, delete, nothing).
- Preprocessing: resizing, normalization, train/val/test split.

 ###Usage
 # Training
python src/train.py

# Evaluation
python src/evaluate.py

# Prediction (single image or webcam)
python src/predict.py --image path/to/image.jpg
Outputs include accuracy metrics, confusion matrix, and random batch visualizations.

🏋️ Training the Model
- Architecture: CNN with Conv2D, MaxPooling, Dropout, and Dense layers.
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy


🔍 Testing / Inference
- Evaluate performance on the test set with evaluate.py.
- Generates:
- Confusion matrix
- Classification report (precision, recall, F1-score)
- predict.py supports webcam inference for real-time gesture recognition.

📊 Results
- Accuracy: ~95% on validation set
- Confusion Matrix: Shows strong performance across most classes, with minor confusion between visually similar gestures.
- Demo: Random batch visualization included in predict.py for beginner-friendly exploration.

<img width="1200" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5866063d-7fcd-4863-825d-c889d7f7c1d6" />
<img width="1500" height="600" alt="prediction1" src="https://github.com/user-attachments/assets/4fa59906-9615-4021-8bb0-2fc5135e9819" />
<img width="1500" height="600" alt="prediction_2" src="https://github.com/user-attachments/assets/82f651eb-3dc7-44ce-90b5-de8bc7c18991" />


## 🚀 Future Work
- Extend recognition to dynamic gestures (video sequences).
- Deploy as a web or mobile app for accessibility.
- Experiment with transfer learning using pretrained CNNs.
- Integrate with sign language → text/speech systems.

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.








