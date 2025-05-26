import os
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image


def predict_image(model, img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0][0]
        label = "Malignant" if prediction >= 0.6 else "Benign"
        return label, float(prediction)
    except Exception as e:
        return "Error", str(e)


def main(model_path, image_dir):
    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        return

    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print("No valid images found in the directory.")
        return

    for img_path in image_files:
        label, confidence = predict_image(model, img_path)
        print(f"{os.path.basename(img_path)} → {label} (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Melanoma Inference Script")
    parser.add_argument("--model", type=str, required=True, help="Path to .keras model file")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    args = parser.parse_args()

    main(args.model, args.dir)
