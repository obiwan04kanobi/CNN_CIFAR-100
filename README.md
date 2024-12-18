# CIFAR-100 Image Classification using CNN

This project involves building and optimizing a Convolutional Neural Network (CNN) to classify images from the CIFAR-100 dataset into 100 fine-grained categories. The model leverages advanced deep learning techniques, such as SE (Squeeze-and-Excitation) blocks, Batch Normalization, and Data Augmentation, to enhance classification accuracy.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Video Demo](#video-demo)
- [Dataset Overview](#dataset-overview)
- [Model Architecture](#model-architecture)
- [Features and Techniques](#features-and-techniques)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Saved Model Integration](#saved-model-integration)
- [Future Improvements](#future-improvements)
- [Contributions](#contributions)
- [License](#license)

---

## Problem Statement

The CIFAR-100 dataset comprises 32x32 color images across 100 categories, ranging from animals and plants to everyday objects. The goal of this project is to classify these images using a robust and optimized deep learning model. 

Key objectives include:
- Designing a scalable CNN architecture.
- Using SE blocks for channel recalibration.
- Implementing data augmentation for better generalization.
- Achieving improved accuracy using batch normalization and dropout.

---

## Video Demo


[Screen Recording 2024-12-18 at 2.14.18 PM.mov](https://github.com/user-attachments/assets/0207ba46-fcf5-4989-9c37-f0466a8de5b7)




---

## Dataset Overview

- **Training Samples**: 50,000 images
- **Test Samples**: 10,000 images
- **Image Dimensions**: 32x32 pixels with 3 color channels (RGB)
- **Classes**: 100 distinct categories, such as 'apple', 'bicycle', 'lion', etc.

The dataset is publicly available and a standard benchmark for image classification tasks.

---

## Model Architecture

The CNN architecture designed for this project includes:
1. **Convolutional Layers**: For feature extraction, with increasing complexity in subsequent layers.
2. **SE (Squeeze-and-Excitation) Blocks**: To dynamically recalibrate channel importance.
3. **Batch Normalization**: For faster convergence and stable training.
4. **Pooling Layers**: MaxPooling layers to reduce spatial dimensions.
5. **Fully Connected Layers**: Dense layers for classification.
6. **Dropout Layers**: For regularization to prevent overfitting.

---

## Features and Techniques

1. **Data Augmentation**: Random transformations (flipping, rotation) applied to training data to enhance diversity.
2. **SE Blocks**: A feature recalibration mechanism to improve learning of important features.
3. **Batch Normalization**: Standardizes activations in intermediate layers for better training dynamics.
4. **Learning Rate Scheduling**: An exponential decay schedule for optimal training convergence.
5. **Saved Model Format**: The trained model is saved in `.keras` format for deployment readiness.

---

## Results

The final model achieved the following metrics on the CIFAR-100 dataset:
- **Test Accuracy**: ~66%
- **Test Loss**: ~1.64

### Visualization
Sample predictions are visualized, showcasing the model's confidence in the assigned class labels.

---

## Setup and Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.0+

### Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/obiwan04kanobi/CNN_CIFAR-100.git
   cd CNN_CIFAR-100
   ```

5. Test the saved model with an image:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('saved_models/cifar100_67_model.keras')
   ```

### Predicting on a New Image
Save an image in the project directory and run:
```python
from tensorflow.keras.preprocessing import image
img = image.load_img('your_image.png', target_size=(32, 32))
img_array = image.img_to_array(img) / 255.0
img_array = img_array[np.newaxis, ...]
prediction = model.predict(img_array)
print(f"Predicted class: {classes[np.argmax(prediction)]}")
```

---

## Saved Model Integration

The trained `.keras` model can be easily integrated into:
1. **Django/Flask Web Applications**: Use TensorFlow's APIs to load the model and serve predictions.
2. **Mobile Applications**: Convert the model to TensorFlow Lite (`.tflite`) for efficient on-device inference.
3. **Cloud Deployment**: Deploy the model as a REST API using platforms like AWS, Azure, or GCP.

---

## Future Improvements

1. **Accuracy Optimization**: Experiment with deeper networks or pre-trained models (transfer learning).
2. **Hyperparameter Tuning**: Fine-tune learning rates, dropout rates, and layer configurations.
3. **Custom Loss Functions**: Tailor loss functions to improve classification performance.
4. **Interactive Web Application**: Build a live web app for real-time predictions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- **CIFAR-100 Dataset**: [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Squeeze-and-Excitation Networks**: [Hu et al., 2018](https://arxiv.org/abs/1709.01507)
- TensorFlow and Keras community for providing tools to build and evaluate the model.
