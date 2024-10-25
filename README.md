# Handwritten Digit Recognition Using Support Vector Machines

This project implements handwritten digit recognition using Support Vector Machines (SVM) on the MNIST dataset. The goal is to classify images of handwritten digits (0-9) into their corresponding digit labels.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview

The MNIST dataset contains 70,000 images of handwritten digits, where each image is a 28x28 pixel grayscale image. This project uses Scikit-learn to implement an SVM model to classify the digits. The model is trained on a subset of the data, evaluated for accuracy, and the performance is reported using a confusion matrix and classification report.

## Dataset

- **Source**: The dataset can be accessed via Scikit-learn using the `fetch_openml` function.
- **Size**: 60,000 training images and 10,000 testing images.
- **Format**: Each image is labeled with the corresponding digit (0-9).

## Requirements

This project requires the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (optional, for visualization)

You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Installation

1. Clone this repository or download the Jupyter notebook file.

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Ensure you have the required libraries installed as mentioned above.

## Usage

1. Open the Jupyter notebook `Handwritten Digit Recognition Using Support Vector Machines.ipynb`.
2. Run each cell in the notebook to load the dataset, preprocess the data, train the SVM model, and evaluate its performance.

### Code Example

Here’s a brief overview of how to load the dataset and train the SVM model:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Results

After training the model, the performance metrics, including accuracy, precision, recall, and F1 score, are printed out. A confusion matrix is also generated to visualize the classification results.

## Conclusion

This project demonstrates how to implement handwritten digit recognition using SVM on the MNIST dataset. The model achieves satisfactory accuracy and can be further improved through hyperparameter tuning and advanced techniques.
