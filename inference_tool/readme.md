## Usage of `inference_tool`

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

> eg: python app.py --model /home/cengo/PycharmProjects/MelonomaDetection/models/final_finetuned_model.keras --dir /home/cengo/PycharmProjects/MelonomaDetection/inference_tool/images


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
Running inference_tool...
Loading model: /home/cengo/PycharmProjects/MelonomaDetection/models/final_finetuned_model.keras
Scanning image directory: /home/cengo/PycharmProjects/MelonomaDetection/inference_tool/images

Processing image1.jpg... Prediction: Malignant (Confidence: 0.85)
Processing image2.png... Prediction: Benign (Confidence: 0.92)
Processing image3.jpg... Prediction: Malignant (Confidence: 0.67)
...
Inference completed for all images.
```
 
 
