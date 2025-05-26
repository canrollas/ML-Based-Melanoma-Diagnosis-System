
# ML-Assisted Melanoma Diagnosis System

## Project Overview

This project focuses on the development of an Machine Learning (ML) system designed to assist in the diagnosis of melanoma. Utilizing the comprehensive ISIC (International Skin Imaging Collaboration) database, the system aims to analyze dermatoscopic images and provide supportive diagnostic information.

## Objective

The primary goal is to leverage modern machine learning techniques to build a robust model capable of differentiating between benign and malignant skin lesions, thereby aiding clinicians and potentially improving diagnostic accuracy and efficiency.

## Context

This initiative serves as a platform to address real-world challenges in medical imaging. It is designed to apply and evaluate contemporary machine learning methodologies in the critical domain of dermatological cancer detection. The project emphasizes practical problem-solving and the application of state-of-the-art AI approaches to medical data.

## Scope (Initial Focus)

The current phase typically involves:

* **Data Acquisition and Preprocessing:** Sourcing and preparing images and metadata from the ISIC database.
* **Model Development:** Designing, training, and validating deep learning models (e.g., Convolutional Neural Networks - CNNs) for image classification.
* **Evaluation:** Assessing model performance using appropriate metrics and validation strategies.

# Melanoma Classification Project - Technical Overview

## 1. Dataset Collection

This project begins with an extensive data collection process, implemented within the `data_retriever/` directory. Two main strategies were employed:

* **ISIC CLI Tool**: Used to fetch dermoscopic images directly from the ISIC Archive.
* **ZIP URL Resources**: Additional curated image datasets were downloaded via public ZIP URLs.

In total, approximately **11,000 images** were collected. The detailed distribution of the images by class (Benign vs Malignant), year, and source is provided in the `README.md` inside the `data_retriever/` directory.
️
> ️️Reminder important note ⚠️ Warning : I used python version of 3.9.6 you can use higher but it can crash due to tensorflow setuptools removed on 3.12 
> You can install for test env with pip install -r requirements.txt 

## 2. Model Training and Evaluation

Training workflows and experimentation are documented across several Jupyter notebooks:

### >Navigating the Notebooks

If you wish to explore the training process and model analysis in detail, we recommend starting by reading the `README.md` located in the **notebooks directory**. It provides an overview of the entire project structure and guides you through the workflow.

The notebooks are organized into subdirectories that reflect each phase of the project:

* **`initial_model/`**
  This folder contains the initial training experiments where a base model was trained **without any fine-tuning**. It establishes a performance baseline and helps identify limitations that motivated further improvements.

* **`weak_decision_investigation/`**
  This directory focuses on analyzing **ambiguous predictions**, specifically those with confidence levels in the range of **0.45 < x < 0.55**. These borderline cases are examined to understand the weaknesses of the initial model and to provide insight into why certain decisions were uncertain.

* **`fine_tune_process/`**
  Here, we applied **hyperparameter tuning** using a **random search** strategy. The goal was to optimize model performance and reduce overfitting observed in earlier stages. This notebook demonstrates the tuning process and presents the selected optimal configuration.

* **`gradcam_process/`**
  To offer interpretability and clinical insight, this folder includes **Grad-CAM-based visualizations** that show which regions the model focuses on when making predictions. These visual cues can help medical professionals assess the model's decision rationale and identify whether it's attending to medically relevant features.

* **`metrics_evaluation/`**
  This final notebook evaluates the trained model using various performance metrics such as **F1-score**, **ROC-AUC**, **precision**, and **recall**. Visualizations and classification reports are also provided to give a comprehensive view of the model’s diagnostic ability.

Each of these stages builds upon the previous one, resulting in a robust pipeline for training, tuning, interpreting, and evaluating a deep learning model for melanoma classification.
 


### Initial Training Phase

* **2,000 images** were initially used to train a base model.
* This model showed high test accuracy (\~95%) but demonstrated clear signs of **overfitting**, as evidenced by divergence in training vs validation performance.


### Expansion and Fine-Tuning

* The dataset was expanded to 11,000 images to reduce overfitting.
* **EfficientNetB0** (with ImageNet weights) was used as the base architecture.
* A random search hyperparameter tuning strategy was employed to optimize:

  * Learning rate
  * Dropout rate
  * Number of epochs

This yielded a final model achieving:

* **Accuracy**: \~86% on the test set
* **Precision / Recall / F1-Score (per class)**:

```
              precision    recall  f1-score   support

      Benign       0.86      0.86      0.86       948
   Malignant       0.86      0.86      0.86       942

    accuracy                           0.86      1890
   macro avg       0.86      0.86      0.86      1890
weighted avg       0.86      0.86      0.86      1890
```

### Explainability and Weak Decision Analysis

The Grad-CAM notebook provides insight into the model's focus areas during prediction:

* **False Negative Analysis**: Cases where malignant tumors were misclassified as benign were examined. Some of these had visual ambiguity or resembled benign lesions.
* **False Positive Analysis**: Benign lesions falsely predicted as malignant often had irregular textures or high-contrast regions.
* In some cases, mislabeling or clinically ambiguous cases (e.g., histologically malignant but visually benign) were detected.

Visualizations include:

* Grad-CAM overlays
* Confusion matrix
* PR and ROC curves

## 3. Inference Tool

A simple and effective inference script is located in the `inference_tool/` directory.

* You can run the script by providing:

  * The path to the test image
  * The path to the trained `.keras` model

Detailed usage instructions are included in the README of the `inference_tool/` directory. This script outputs the prediction label and confidence score.

## 4. Models Directory

This folder (`models/` )  contains all the final models:

* `base_model.keras`: Pretrained but **not** fine-tuned model
* `finetuned_model.keras`: Final model after hyperparameter optimization and training on the full dataset


 

This project aims to strike a balance between model accuracy and interpretability in a medically critical task like melanoma detection. The integration of Grad-CAM visualization with a solid training strategy provides valuable insights not only into the performance but also the trustworthiness of the model.

> For questions or reproducibility details, please check the respective README files in each directory.









