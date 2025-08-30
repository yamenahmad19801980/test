# AI-Driven Water Leakage Detection Simulation

This repository contains a Python script `simulate_and_train.py` that simulates IoT sensor data for
water distribution networks and trains a time-series neural network (LSTM) to detect leakages.

## Features
- Generates synthetic data for normal and leaking scenarios (small, medium, catastrophic).
- Trains an LSTM-based classifier using TensorFlow/Keras.
- Provides confusion matrix and ROC AUC evaluation.
- Includes a simple example for leak localization based on pressure drops.

## Usage
```bash
python simulate_and_train.py
```

The script will create a dataset `synthetic_sensors.csv`, train the model, and output evaluation
metrics. TensorFlow and required dependencies must be installed.

## License
MIT
