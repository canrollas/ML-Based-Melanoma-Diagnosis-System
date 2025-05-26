## Project Workflow and Methodology

### 1. Initial Model Development with a Baseline Dataset

The project began with the construction of a baseline convolutional neural network (CNN) model for melanoma classification using a dataset consisting of 2,000 dermatoscopic images. This dataset included benign and malignant skin lesion samples. Although the initial training results were promising, it became evident that the dataset size was insufficient to achieve clinically meaningful generalization. Overfitting and limited diversity in skin lesion patterns indicated the necessity of a more comprehensive dataset.

### 2. Dataset Expansion and Collection Strategy

To improve the model’s robustness, the dataset was expanded by incorporating approximately 11,000 images from the ISIC (International Skin Imaging Collaboration) archive. This dataset included a balanced representation of both benign and malignant cases and covered various acquisition conditions, skin tones, and lesion types. Image augmentation and preprocessing techniques such as resizing, normalization, and augmentation (rotation, flipping) were applied to enhance data variability during training.

### 3. Fine-Tuning Strategy and Transfer Learning

After establishing a stable baseline model, a fine-tuning strategy was formulated to leverage pretrained weights from EfficientNetB0. The initial training phase involved freezing the lower layers of the network to preserve learned generic features. Subsequent stages gradually unfroze deeper layers while using a lower learning rate, allowing the model to adapt to melanoma-specific features without catastrophic forgetting. This step significantly improved performance on validation and test sets, suggesting successful transfer of high-level features.

### 4. Explainability with Grad-CAM Visualizations

To gain insights into the decision-making process of the model, Grad-CAM (Gradient-weighted Class Activation Mapping) was applied to visualize class-specific attention regions. These heatmaps helped assess whether the model was focusing on clinically relevant areas, such as lesion borders, asymmetries, or color variations. Observations indicated that the model’s attention was often centered on the lesion mass in benign cases, whereas attention shifted to irregular edges or peripheral zones in malignant predictions. This stage was critical in validating the interpretability and reliability of the model in medical settings.

### 5. Analysis of Weak Decisions and Failure Cases

Following visual inspection, a comprehensive analysis of weak decisions—false positives and false negatives—was conducted. Images with ambiguous patterns, poor lighting, or atypical benign lesions were frequently misclassified. To address this, hypotheses were proposed, including the need for lesion segmentation, attention mechanisms, or ensemble methods to reduce uncertainty. These findings guided potential improvements for future iterations and offered direction for more robust modeling.

### 6. Evaluation and Metrics Reporting

The final phase included a full evaluation of the trained model using a diverse set of metrics suited for binary medical classification tasks. These included:

* Accuracy
* Precision and Recall
* F1-score
* Sensitivity and Specificity
* ROC-AUC score
* Confusion Matrix analysis

Particular emphasis was placed on minimizing false negatives due to their critical clinical consequences. The results showed a well-balanced performance, with a strong ability to detect malignant lesions while maintaining a low false-positive rate. These metrics validated the effectiveness of the training strategy and justified the design decisions taken throughout the project.

 