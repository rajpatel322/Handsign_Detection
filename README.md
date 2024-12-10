# Real-Time Hand Gesture Detection with OpenCV and SVM

This project implements a robust pipeline for detecting and classifying hand gestures in real time using **OpenCV** and **Support Vector Machines (SVM)**. The model is trained on custom hand gesture datasets and achieves high accuracy for gesture recognition.

---

## Table of Contents
1. [Overview](#overview)
2. [Workflow](#workflow)
---

## Overview

This project leverages:
- **OpenCV** for real-time image processing.
- **Mediapipe** for hand detection and landmark extraction.
- **Scikit-learn's SVM** for gesture classification.

---

## Workflow

### 1. Data Collection
- Captured **200 images** for each gesture, representing numbers from **1 to 10**.
- Used **Mediapipe** to detect hands in frames and extract meaningful features (e.g., landmarks) for dataset creation.

### 2. Dataset Preparation
- Stored the processed data in a `.pickle` file for easy serialization and deserialization.
- Labeled gestures to enable supervised learning.

### 3. Model Training
- Utilized **Scikit-learn's Support Vector Machine (SVM)** for classification.
- Split the data into:
  - **80% training** set for learning.
  - **20% testing** set for evaluation.
- Achieved an impressive **98% accuracy** on the test set.

### 4. Real-Time Detection
- Integrated the trained SVM model with **OpenCV** to classify gestures in real time.
- Used live camera feeds to detect and process hand gestures dynamically.

---
