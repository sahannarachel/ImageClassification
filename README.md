# ImageClassification

# Fashion MNIST ConvNet Classifier

This repository contains a Convolutional Neural Network (ConvNet) implemented using TensorFlow and Streamlit for classifying images from the Fashion MNIST dataset.

## Overview

The Fashion MNIST dataset contains 60,000 training images and 10,000 testing images, classified into 10 categories of clothing and accessories. The ConvNet model built in this project is trained on the training set and evaluated on the testing set.

## Requirements

- Python 3.x
- TensorFlow
- Streamlit
- NumPy
- Pandas

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the application locally, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your web browser with the Streamlit application. You can then upload an image and click the "Classify" button to see the model's prediction.

## Model Architecture

The ConvNet model architecture used in this project consists of the following layers:

- Input layer: Accepts images with dimensions (28, 28, 1).
- Convolutional layers: Two convolutional layers with ReLU activation followed by max-pooling layers.
- Flatten layer: Flattens the output of the convolutional layers.
- Dense layers: Two dense layers with ReLU activation.
- Output layer: Dense layer with softmax activation for class probabilities.

## Class Labels

The model predicts the following class labels:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Training

The model is trained for 5 epochs with the Adam optimizer and sparse categorical cross-entropy loss function.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please feel free to open an issue or create a pull request.

