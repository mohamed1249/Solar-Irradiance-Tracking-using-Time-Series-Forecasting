# Solar Irradiance Tracking using Time Series Prediction

This project aims to predict the next hour's solar irradiance data using time series forecasting techniques. By leveraging various machine learning models, including LSTM, KNN, SVR, and ARIMA, the system can provide accurate predictions to optimize solar energy usage.

## Table of Contents
1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
   - [LSTM](#lstm)
   - [Bidirectional LSTM](#bidirectional-lstm)
   - [KNN](#knn)
   - [SVR](#svr)
   - [ARIMA](#arima)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Conclusion](#conclusion)

## Introduction

Solar irradiance prediction is crucial for optimizing solar energy systems. This project utilizes historical solar irradiance data to forecast future values. The approach involves data preprocessing, feature engineering, and applying various machine learning models to achieve the best prediction accuracy.

## Technologies Used

- **Python**
- **Jupyter Notebook**
- **Node-RED**
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, Keras, Statsmodels

## Data Preprocessing

1. **Importing Libraries**: Essential libraries for data manipulation, visualization, and modeling are imported.
2. **Reading Data**: The dataset (`Hourly-2019.csv`) is read and basic exploratory data analysis (EDA) is performed.
3. **Feature Engineering**: Relevant lag features are created based on autocorrelation and partial autocorrelation analysis.
4. **Data Scaling**: Features and target variables are scaled using `StandardScaler` for better model performance.

## Modeling

### LSTM

1. **Model Architecture**: A sequential model with LSTM layers is built.
2. **Training**: The model is trained with a fixed number of epochs.
3. **Prediction & Evaluation**: Model performance is evaluated using MAE, MSE, RMSE, and R² metrics.

### Bidirectional LSTM

1. **Model Architecture**: A sequential model with Bidirectional LSTM layers.
2. **Training**: Similar to LSTM, but with slightly more epochs for improved accuracy.
3. **Evaluation**: The model is evaluated using the same metrics.

### KNN

1. **Hyperparameter Tuning**: Grid search is performed to find the optimal number of neighbors.
2. **Training & Prediction**: The KNN model is trained and predictions are made.
3. **Evaluation**: Performance metrics are calculated.

### SVR

1. **Hyperparameter Tuning**: Grid search is used to find the best kernel, C, coef0, and gamma parameters.
2. **Training & Prediction**: The SVR model is trained and predictions are made.
3. **Evaluation**: Performance metrics are calculated.

### ARIMA

1. **Model Building**: ARIMA model parameters (p, d, q) are determined from ACF and PACF plots.
2. **Training & Prediction**: The model is trained and predictions are made.
3. **Evaluation**: Performance metrics are calculated.

## Evaluation

The models are evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R² (R-Squared)**

## Results

The performance of each model is visualized using bar plots for RMSE and MAE, allowing for a clear comparison of the models' effectiveness.

## How to Run

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/mohamed1249/Solar-Irradiance-Tracking-using-Time-Series-Forecasting
   ```
2. **Navigate to the Project Directory**:
   ```sh
   cd [project directory]
   ```
3. **Install Required Libraries**:
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Jupyter Notebook**:
   ```sh
   jupyter notebook
   ```
5. **Explore the Project**: Open and run the notebook to see the full workflow and results.

## Conclusion

This project demonstrates how different machine learning models can be applied to time series data for predicting solar irradiance. By comparing multiple models, we can select the most accurate one for practical applications in solar energy optimization.
