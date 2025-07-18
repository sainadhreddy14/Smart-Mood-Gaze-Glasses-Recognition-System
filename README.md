# Smart Mood, Gaze & Glasses Recognition System

This project provides a user-friendly interface to detect a person's mood (Angry, Happy, Neutral), gaze direction (Center, Right, Left), and whether they are wearing glasses using a webcam. It uses TensorFlow/Keras for machine learning, OpenCV for image processing, and Tkinter for the GUI.

## Features
- Train a new model by capturing and labeling images for all 18 combinations of mood, gaze, and glasses.
- Test trained models on live webcam feed or uploaded images.
- Save and manage datasets, models, and training logs.
- Clear instructions and progress feedback in the GUI.

## Project Structure
```
project/
├── dataset/
│   ├── combination_1/
│   ├── combination_2/
│   └── ...
├── models/
│   ├── model_YYYYMMDD_HHMMSS.h5
│   └── ...
├── training_logs/
│   ├── model_YYYYMMDD_HHMMSS/
│   │   ├── images/
│   │   ├── metrics.csv
│   │   └── plots/
├── main.py
├── requirements.txt
└── README.md
```

## Setup
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the application:
   ```
   python main.py
   ```

## Usage
- **Train New Model**: Follow on-screen instructions to capture images for each combination. Images are saved in `dataset/`.
- **Test Model**: Select a trained model from the dropdown and test on webcam or images.
- Training logs and metrics are saved in `training_logs/`.

## Requirements
- Python 3.7+
- Webcam for image capture

## Libraries Used
- TensorFlow/Keras
- OpenCV
- Tkinter
- NumPy
- Matplotlib
- Pillow

## Author
- Your Name

## Demo & Screenshots

Below are some screenshots demonstrating the application's features and user interface. Please add your images in the spaces provided:

### Main Menu
![Main Menu Screenshot](path/to/main_menu_screenshot.png)
*Description: The main menu allows you to capture images, train a new model, test a model, clear all data, or quit the application.*

### Model Testing Interface
![Model Testing Screenshot](path/to/model_testing_screenshot.png)
*Description: The model testing interface lets you select a trained model, choose between webcam or image input, and displays the prediction result with the input image.*