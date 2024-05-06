import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# Define helper functions
def create_convolutional_layer(filters, kernel_size, activation='relu'):
    return layers.Conv2D(filters, kernel_size, activation=activation)

def create_maxpooling_layer(pool_size=(2, 2)):
    return layers.MaxPooling2D(pool_size)

def create_dense_layer(units, activation='relu'):
    return layers.Dense(units, activation=activation)

# Build ConvNet model

def build_convnet(input_shape, num_classes):
    # Define the input layer
    inputs = Input(shape=input_shape)

    # Convolutional layers
    conv1 = create_convolutional_layer(32, (3, 3), activation='relu')(inputs)
    pool1 = create_maxpooling_layer()(conv1)
    conv2 = create_convolutional_layer(64, (3, 3), activation='relu')(pool1)
    pool2 = create_maxpooling_layer()(conv2)

    # Flatten layer
    flatten = layers.Flatten()(pool2)

    # Dense layers
    dense1 = create_dense_layer(128, activation='relu')(flatten)
    outputs = create_dense_layer(num_classes, activation='softmax')(dense1)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Define the model globally
input_shape = (28, 28, 1)
num_classes = 10
model = build_convnet(input_shape, num_classes)

# Define mapping between class indices and labels
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Streamlit UI
st.title('Fashion MNIST ConvNet Classifier')

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Function to train the model
def train_model():
    # Train the model
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    st.write("Test accuracy:", test_acc)

# Check if the "Classify" button is pressed
if st.button('Classify'):
    # Train the model (optional, remove this line if you want to skip training on each click)
    train_model()

    # Predict the class for the uploaded image
    if uploaded_image is not None:
        image = tf.image.decode_image(uploaded_image.read(), channels=1)
        image = tf.image.resize(image, [28, 28])
        image = np.expand_dims(image, axis=0) / 255.0

        st.image(image.reshape((28, 28)), caption='Uploaded Image', use_column_width=True)

        prediction = model.predict(image)
        predicted_class_index = np.argmax(prediction)
        predicted_label = class_labels.get(predicted_class_index, "Unknown")
        st.write("Prediction:", predicted_label)
