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
- [License](#license)

## Overview

The MNIST dataset contains 70,000 images of handwritten digits, where each image is a 28x28 pixel grayscale image. This project uses Scikit-learn to implement an SVM model to classify the digits. The model is trained on a subset of the data, evaluated for accuracy, and the performance is reported using a confusion matrix and classification report.

## Dataset

- **Source**: The dataset is accessed using Scikit-learn's `fetch_openml` function.
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

1. Clone this repository:

   ```bash
   git clone https://github.com/Vishnugnath/Handwritten-Digit-Recognition-Using-Support-Vector-Machines.git
   cd Handwritten-Digit-Recognition-Using-Support-Vector-Machines
   ```

2. Ensure you have the required libraries installed as mentioned above.

## Usage

1. Open the Jupyter notebook `Handwritten Digit Recognition Using Support Vector Machines.ipynb`.
2. Run each cell in the notebook to:
   - Load the dataset
   - Preprocess the data
   - Train the SVM model
   - Evaluate the model's performance

## Results

After training the model, performance metrics including accuracy, precision, recall, and F1 score are calculated. A confusion matrix is also generated to visualize the classification results, providing insights into the accuracy of each digit classification.

## Conclusion

This project demonstrates how to implement handwritten digit recognition using SVM on the MNIST dataset. The SVM classifier achieves satisfactory accuracy and can be further improved through hyperparameter tuning and more advanced techniques like deep learning.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
