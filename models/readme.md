# Load and Fine-Tune a Saved Melanoma Classification Model

This document explains how to load a pre-trained EfficientNetB0 model saved in  `.keras` format and fine-tune it using additional data or epochs.

---

## Load the Model

Make sure the model file is in your Google Drive or local path.

```python
from tensorflow.keras.models import load_model

# Update this path to the location of your model
model_path = "./efficientnetb0_melanoma_model.h5"

# Load the saved model
model = load_model(model_path)
model.summary()  # Optional: view model architecture
