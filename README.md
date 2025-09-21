Air Quality Forecasting
Project Overview

This project focuses on forecasting air pollution levels, particularly PM2.5 concentrations, in Beijing using historical air quality and weather data. It is part of the Machine Learning Techniques I course and aims to apply Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models to a real-world problem. Accurate predictions of PM2.5 concentrations can help governments and communities take timely measures to mitigate public health risks and plan urban interventions.

The main objectives of this project are:

Preprocess sequential air quality and weather data to uncover meaningful patterns.

Design, train, and evaluate RNN/LSTM models for PM2.5 prediction.

Fine-tune the model and run multiple experiments to optimize performance, with a target Root Mean Squared Error (RMSE) below 4000 on the Kaggle leaderboard.

## Dataset

The dataset for this project is provided through the Kaggle competition and includes:

train.csv: Historical data used for training the model.

test.csv: Data for which predictions are required.

sample_submission.csv: Submission format for Kaggle.

The dataset contains air quality readings, meteorological data, and temporal features. Preprocessing steps include handling missing values, feature scaling, and creating sequential data for model input.

## Model Design

The primary model used in this project is an LSTM network. The architecture that achieved the best performance includes:

Input sequence length: 48 timesteps

LSTM layers: 64 → 32 units

Dropout: 0.2 to reduce overfitting

Batch size: 128

Learning rate: 0.001

The model was trained on historical sequences of 17 features, capturing both air quality and weather parameters. Early stopping and multiple experiments were used to fine-tune the model for optimal RMSE.

## Experiments

A series of experiments were conducted to evaluate model performance with different hyperparameters and sequence lengths. Key parameters include:

Sequence length (24 vs. 48)

LSTM units (32, 64)

Dropout rates (0.1, 0.2)

Batch size (64, 128)

Learning rate (0.001, 0.0005)

Number of epochs

The results were summarized in a table to compare the effect of each configuration on validation RMSE and model stability.

## Results

The final LSTM model achieved a validation RMSE of approximately 75.18 with a sequence length of 48 timesteps, outperforming shorter sequences. Key observations:

Longer sequences better capture temporal dependencies and improve peak prediction.

Dropout helped reduce overfitting without significantly impacting convergence.

Model performance is sensitive to batch size and learning rate adjustments.

## Usage Instructions
Installation

Clone the repository:

git clone <your-github-repo-link>
cd air-quality-forecasting


## Install dependencies:

pip install -r requirements.txt

Running the Project

Load and preprocess the dataset:

# Example
from data_preprocessing import preprocess_data
X_train, y_train, X_test = preprocess_data('train.csv', 'test.csv')


Train the model:

from models import train_lstm
model = train_lstm(X_train, y_train, epochs=12, batch_size=128)


Make predictions on test data:

predictions = model.predict(X_test)
# Format according to sample_submission.csv


Submit predictions to Kaggle.

Contributing

Contributions are welcome. If you find bugs or want to improve the model, please submit a pull request or open an issue.

License

This project is for academic purposes and submitted as part of the Machine Learning Techniques I course. Please credit the author when using or modifying this work.

References

Kaggle Air Quality Forecasting Competition

TensorFlow/Keras documentation for LSTM models

Course materials from Machine Learning Techniques I
