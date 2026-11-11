# Plant Disease Detection Using Deep Learning

A deep learning-based application that identifies plant diseases from leaf images using a Convolutional Neural Network (CNN). The project uses TensorFlow and Keras for model training and Streamlit to provide an interactive interface for disease prediction.

## Features

- Upload a plant leaf image for disease detection.
- Capture an image using the camera.
- Classify images into 10 tomato leaf categories.
- Display the predicted disease and confidence score.
- Provide basic disease descriptions and management suggestions.

## Technologies Used

- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- Streamlit
- NumPy
- Pillow

## Model Details

- Model architecture: CNN
- Image dimensions: 224 × 224 pixels
- Batch size: 32
- Training epochs: 10
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Dataset split: 80% training and 20% validation

## Model Performance

| Metric | Accuracy |
|---|---:|
| Training Accuracy | 96.75% |
| Final Validation Accuracy | 90.84% |
| Best Validation Accuracy | 92.15% |

*These are training and validation results, not independent test-set accuracy.*

## Project Structure

```text
plant-disease-detection/
├── app.py
├── train_model.py
├── model_results.py
├── requirements.txt
├── README.md
└── model/
    ├── class_names.txt
    └── training_history.json
```

## How to Run

1. Clone the repository:

   `git clone https://github.com/aishwaryasharma518/plant-disease-detection.git`

2. Navigate to the project directory:

   `cd plant-disease-detection`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Ensure the trained model is available at:

   `model/plant_disease_model.h5`

5. Run the Streamlit application:

   `streamlit run app.py`

## Note About the Trained Model

The trained model file (`plant_disease_model.h5`) is not included in this repository because of its large file size (127 MB). The application requires this file to perform predictions.

## Author

**Aishwarya Sharma**

Final Year B.Tech Computer Science Engineering Student
