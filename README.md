# Rice-CNN-Classifier
This project is a Convolutional Neural Network (CNN) based classifier trained on rice images to automatically predict rice varieties. The model is built with TensorFlow/Keras, trained on Kaggle, and deployed with Streamlit for an interactive web app.

🌾 Rice CNN Classifier

A deep learning project to classify rice grain images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The project includes a Streamlit-based interactive interface for predicting rice types from images.

📌 Overview

This project demonstrates:

Image classification using CNNs

Data preprocessing and augmentation

Model evaluation and visualization

Deployment with Streamlit for user-friendly interaction

The CNN model is trained on a rice image dataset and can predict among five rice types: Basmati, Jasmine, Arborio, Ipsala, and Karacadag.

🧪 Requirements

Python 3.8+

TensorFlow

Keras

Streamlit

NumPy

Matplotlib

PIL (Pillow)

You can install dependencies with
`pip install -r requirements.txt`

📂 Project Structure

```
Rice-CNN-Classifier/
├── app.py                  # Streamlit app for predictions
├── rice_cnn_model.h5       # Trained CNN model
├── notebook/               # Jupyter notebooks for training and analysis
├── data/                   # Rice image dataset (download manually)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```
🚀 Training the Model

1.Load dataset

Organize images in data/ folder with subfolders per rice type.

2.Data preprocessing & augmentation

`datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)`

3.Build CNN model
```
model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(64,64,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(5, activation="softmax")
])
```
4.Compile & train

```
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_history = model.fit(train_data, epochs=10, validation_data=test_data)
```
5.Save the trained model
`model.save("rice_cnn_model.h5")`

6.Evaluate model

```
loss, acc = model.evaluate(test_data)
print(f"Validation Accuracy: {acc*100:.2f}%")
```

🧑‍💻 Using the Streamlit App

Run the app:
`streamlit run app.py`


Features:

Upload an image of a rice grain

Predict the rice type

Display probabilities with a bar chart highlighting the predicted class

📊 Model Performance
The model achieves high accuracy on the validation set.

Training and validation loss/accuracy plots are generated for analysis.

📥 Dataset
The rice dataset can be downloaded from Kaggle:

https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

Organize the images in subfolders per rice type inside data/ before training
