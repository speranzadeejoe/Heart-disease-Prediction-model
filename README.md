# Heart-disease-Prediction-model
# Heart Disease Prediction Model

## Overview
This project is a binary classification model designed to predict the likelihood of heart disease based on given input features. It is implemented using TensorFlow and Keras, utilizing a simple feedforward neural network.

## Dataset
The model assumes an input dataset with **13 features** (as specified in the `input_dim=13` parameter). These features could include common health indicators such as:
- Age
- Sex
- Chest pain type
- Blood pressure
- Cholesterol levels
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression induced by exercise
- Slope of the peak exercise ST segment
- Number of major vessels colored by fluoroscopy
- Thalassemia

The target variable (`y`) is binary:
- `0` for no heart disease
- `1` for heart disease presence

## Model Architecture
The model is a simple **feedforward neural network** with:
1. **Input layer**: 13 neurons (one for each feature)
2. **Hidden layer**: 11 neurons with ReLU activation
3. **Output layer**: 1 neuron with Sigmoid activation (for binary classification)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(11, activation="relu", input_dim=13))  # Hidden layer
model.add(Dense(1, activation="sigmoid"))  # Output layer
```

## Compilation & Training
The model is compiled using:
- **Loss function**: `binary_crossentropy` (since it is a binary classification problem)
- **Optimizer**: `adam`
- **Metric**: `accuracy`

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.22, epochs=300)
```

## Training Details
- **Validation split**: 22% of data used for validation
- **Epochs**: 300 (for better convergence)
- **Optimizer**: Adam (for adaptive learning rate)

## Installation & Requirements
To run this model, install the following dependencies:

```bash
pip install tensorflow numpy pandas matplotlib
```

## Usage
1. Prepare your dataset (`X` and `y`).
2. Run the script to train the model.
3. Use the trained model to make predictions:

```python
predictions = model.predict(new_data)
```

## Results & Performance
The model's performance should be evaluated using accuracy, precision, recall, and F1-score. You can plot the training loss and accuracy to analyze convergence.

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Future Improvements
- Try adding more layers to improve model complexity.
- Experiment with dropout layers to prevent overfitting.
- Perform hyperparameter tuning for better optimization.
- Test different activation functions.

## Conclusion
This is a basic deep learning model for heart disease prediction. With further optimizations and feature engineering, it can be improved for real-world applications.

---
