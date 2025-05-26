## Usage of `inference_tool`


---

### ⚠️ Warning

This tool is for research and educational purposes only. When testing the model, **select images with clearly visible asymmetry, irregular borders, or dark pigmentation patterns**—features typically associated with malignant cases.
Please note that **most images labeled as "malignant" online are diagnosed via biopsy**, even when they visually resemble benign lesions. Therefore, model predictions may **not align with medical ground truth** without clinical context. Use with caution and do not rely on this tool for real-world diagnostic decisions.

---

The `inference_tool` (executed via `app.py`) is a command-line utility designed to perform inference using a pre-trained deep learning model. Its primary purpose is to classify skin lesions from images, predicting whether they are benign or malignant. This tool allows for rapid testing and application of your trained model on single or multiple images.

### Prerequisites

Before running the tool, ensure you have the necessary Python libraries installed. Typically, a project includes a `requirements.txt` file. If so, you can install the dependencies using:
```bash
pip install -r requirements.txt
```
It is also recommended to have Python 3.x installed.

### Basic Usage

come to this directory from terminal. run it like this
> path_to_model_file : in the main directory, you can use the models dir. It includes keras files.

> path_to_image_directory : you can use the images dir inside of this directory. It includes many images to test.

> eg: python app.py --model ../models/final_finetuned_model.keras --dir trial


```bash
python app.py --model <PATH_TO_MODEL_FILE> --dir <PATH_TO_IMAGE_DIRECTORY>
```

### Command-Line Arguments

* **`--model <PATH_TO_MODEL_FILE>`** (Required)
    * **Description**: Specifies the full path to the pre-trained Keras model file that will be used for inference.
    * **Supported Formats**: Typically `.keras` (recommended) or `.h5` model files.
    * **Example**:
        ```bash
        --model /home/cengo/PycharmProjects/MelonomaDetection/models/final_finetuned_model.keras
        ```

* **`--dir <PATH_TO_IMAGE_DIRECTORY>`** (Required)
    * **Description**: Specifies the full path to the directory containing the image(s) you want to classify. The tool will process all supported image formats (e.g., JPG, PNG) found within this directory.
    * **Note**: It is expected that this directory contains only the images intended for inference. The handling of subdirectories depends on the tool's implementation.
    * **Example**:
        ```bash
        --dir /home/cengo/PycharmProjects/MelonomaDetection/inference_tool/images
        ```

### Expected Output

When executed, the `inference_tool` typically performs the following steps:

1.  Loads the specified pre-trained model.
2.  Preprocesses each image found in the provided directory according to the model's requirements.
3.  Performs inference on each preprocessed image to obtain a prediction.
4.  Outputs the prediction (e.g., "Malignant" or "Benign") for each processed image. This output is usually printed to the console, often including the image filename and a confidence score associated with the prediction.

**Illustrative Console Output Example:**
```
1/1 ━━━━━━━━━━━━━━━━━━━━ 6s 6s/step
malignant_ISIC_3985031.jpg → Malignant (Confidence: 0.98)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 510ms/step
img_1.png → Benign (Confidence: 0.09)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 513ms/step
img_2.png → Benign (Confidence: 0.39)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 510ms/step
benign_3.png → Benign (Confidence: 0.25)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 536ms/step
benign_2.png → Benign (Confidence: 0.58)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 512ms/step
malignant_2.png → Malignant (Confidence: 0.93)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 519ms/step
img.png → Malignant (Confidence: 0.98)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 515ms/step
malignant_3.png → Benign (Confidence: 0.47)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 512ms/step
malignant.png → Benign (Confidence: 0.53)
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 515ms/step
benign.png → Benign (Confidence: 0.46)

```

 
 
