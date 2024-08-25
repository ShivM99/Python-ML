# FraudLib
FraudLib is a Python library for fraud detection, providing tools and algorithms to identify fraudulent transactions in various domains.

## Features
- Implements a variety of fraud detection algorithms, including:
  - Anomaly detection methods (isolation forest)
  - Artificial neural network
- Provides utilities for data preprocessing, feature engineering, and model evaluation.
- Designed for flexibility and ease of use, suitable for both beginners.

## Cloning the repository
You can clone this repository using the command:
Clone the repository:
   ```bash
   git clone https://github.com/ShivM99/Python-ML/Fraud_Detection.git
   ```

## Installation
You can install FraudLib using pip:
  ```bash
  pip install /path/to/mypackage-1.0.0-py3-none-any.whl
  ```

FraudLib requires Python 3.6 or later.

## Usage
Here's a basic example of how to use FraudLib to detect fraud in a dataset:

```python
# Import FraudLib
import fraudlib

# Analyse data
fraudlib.data_analysis (dataset)

# Preprocess data
preprocessed_data = fraudlib.data_preprocessing (dataset)

# Train an isolation forest anomaly detection model
model = fraudlib.isolationForest (x_train, y_train, x_test, y_test)

# Train an ANN classification model
model = fraudlib.ANN (x_train, y_train, x_test, y_test)
```

## Contributions
Contributions to FraudLib are welcome! If you find a bug or have a feature request, please open an issue on GitHub. If you'd like to contribute code, please fork the repository and submit a pull request.