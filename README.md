#  Ames Housing Price Prediction

A machine learning project that predicts residential property prices using the Ames Housing Dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and deployment through an interactive Streamlit web application.

## 📌 Project Overview

House price prediction is a supervised machine learning regression problem where the goal is to estimate the market value of a property based on its characteristics.

This project uses the Ames Housing Dataset and applies Ridge Regression to build a robust predictive model capable of estimating house sale prices from multiple numerical and categorical features.

## 🎯 Objective

* Analyze housing data to identify important price-driving factors.
* Build and evaluate machine learning regression models.
* Optimize model performance using GridSearchCV.
* Deploy the final model as an interactive Streamlit application.

## 📊 Dataset

The Ames Housing Dataset contains detailed information about residential properties, including:

* Overall Quality
* Living Area
* Basement Area
* Garage Capacity
* Number of Bathrooms
* Neighborhood
* House Style
* Year Built

Target Variable:

* SalePrice

## 🔍 Exploratory Data Analysis

The notebook includes:

* SalePrice distribution analysis
* Correlation analysis of numerical features
* Top correlated feature visualization
* Correlation heatmap of key features
* Insights into factors affecting house prices

## 🤖 Machine Learning Pipeline

### Preprocessing

* Missing value handling
* One-hot encoding for categorical features
* Feature scaling using StandardScaler

### Models Evaluated

* Linear Regression
* Ridge Regression
* Lasso Regression
* Elastic Net

### Hyperparameter Tuning

GridSearchCV was used to identify the optimal model parameters and improve generalization performance.

## 🏆 Final Model

### Ridge Regression

Ridge Regression was selected as the final model because it effectively handles multicollinearity, reduces overfitting, and provides stable predictions.

## 📈 Model Performance

| Metric   | Score    |
| -------- | -------- |
| R² Score | ~0.90    |
| MAE      | ~$15,542 |

### Interpretation

On average, the model's predictions differ from actual house prices by approximately **$15,542**, indicating strong predictive performance for residential property valuation.

## 🌐 Streamlit Application Features

* Interactive property input form
* Neighborhood and house style selection
* Real-time price prediction
* Estimated price range display
* Comparison with dataset averages
* Key factors influencing predicted price
* Model performance metrics display

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Streamlit
* Jupyter Notebook

## 📂 Project Structure

ames-house-price-prediction/
│
├── app.py
├── prediction.ipynb
├── requirements.txt
├── ridge_model.pkl
├── processor.pkl
├── scaler.pkl
├── AmesHousing.csv
└── README.md
```

## 🚀 Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 📄 License

This project is intended for educational and portfolio purposes.
