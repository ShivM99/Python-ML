# FraudLib
FraudLib is a Python library for fraud detection, providing tools and algorithms to identify fraudulent transactions in various domains.

## Features
- Implements a variety of fraud detection algorithms, including:
  - Supervised learning techniques (e.g., logistic regression, random forests)
  - Anomaly detection methods (e.g., isolation forest, one-class SVM)
- Provides utilities for data preprocessing, feature engineering, and model evaluation.
- Designed for flexibility and ease of use, suitable for both beginners and advanced users.
- Well-documented with examples and tutorials to help users get started quickly.

## Installation
You can install FraudLib using pip:
pip install /path/to/mypackage-1.0.0-py3-none-any.whl

FraudLib requires Python 3.6 or later.

## Usage
Here's a basic example of how to use FraudLib to detect fraud in a dataset:

> Import FraudLib
import fraudlib

> Analyse data
fraudlib.data_analysis (dataset)

> Preprocess data
preprocessed_data = fraudlib.data_preprocessing (dataset)

> Train an isolation forest anomaly detection model
model = fraudlib.isolationForest (x_train, y_train, x_test, y_test)

> Train an ANN classification model
model = fraudlib.ANN (x_train, y_train, x_test, y_test)

## Contributions
Contributions to FraudLib are welcome! If you find a bug or have a feature request, please open an issue on GitHub. If you'd like to contribute code, please fork the repository and submit a pull request.