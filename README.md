# CNN-for-LSTM

# CNN-LSTM for Univariate Time Series Forecasting

This project demonstrates how to build and train a hybrid CNN-LSTM model for univariate time series forecasting using Keras with TensorFlow backend.

The approach combines the strengths of 1D Convolutional Neural Networks (CNNs) for feature extraction and Long Short-Term Memory (LSTM) networks for capturing temporal dependencies in sequential data.


📌 # Overview

The code consists of two parts:

1. Simple Integer Sequence Example

  * Uses a toy dataset of sequential integers.

  * The model learns to predict the next value in the sequence.

2. Custom Floating-Point Dataset Example

  * Uses a small dataset of real-valued numbers.

  * Trains the CNN-LSTM model to forecast the next time step.

Both examples reshape the dataset into a format suitable for CNN-LSTM:

[samples, subsequences, timesteps, features]


Where:

samples → number of training examples

subsequences → how each sequence is divided for CNN processing

timesteps → steps per subsequence

features → number of features per timestep (1 in this case since it’s univariate)


⚙️ Model Architecture

The model architecture is defined as:

1. TimeDistributed Conv1D → applies convolution across each subsequence

2. TimeDistributed MaxPooling1D → down-samples extracted features

3. TimeDistributed Flatten → flattens the feature maps for each subsequence

4. LSTM (50 units) → learns sequential dependencies

5. Dense (1 unit) → outputs the prediction



🧩 Example Usage
Training & Prediction

# Example integer dataset
X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([50, 60, 70, 80])

# Reshape input to [samples, subsequences, timesteps, features]
X = X.reshape((X.shape[0], 2, 2, 1))

# Train the model
model.fit(X, y, epochs=500, verbose=0)

# Make prediction
x_input = array([50, 60, 70, 80]).reshape((1, 2, 2, 1))
yhat = model.predict(x_input, verbose=0)
print("Predicted:", yhat)


📊 Key Points

* CNN-LSTM models are useful when your sequence data can be split into smaller subsequences for feature extraction.

* Univariate forecasting means predicting a single variable based on its past values.

* The dataset is very small in this example, intended for demonstration purposes.

* For real-world forecasting, you would use larger datasets and likely tune hyperparameters (filters, kernel size, LSTM units, etc.).


🚀 Requirements

* Python 3.7+

* TensorFlow / Keras

* NumPy

Install dependencies:

pip install tensorflow numpy
