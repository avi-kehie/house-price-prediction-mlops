# House Price Prediction Using MLOps

## 1. Project Overview

This project predicts house prices using Machine Learning and demonstrates a basic MLOps workflow.

The model uses four input features:
- Area (square feet)
- Number of bedrooms
- Number of bathrooms
- House age (years)

A Linear Regression algorithm is used to train the model.

## 2. Technologies Used

Python  :	Main programming language used to develop the project.
Pandas  :	Loads, organises and processes the house-price dataset.
NumPy   :	Generates numerical data and performs mathematical operations.
Scikit-learn    :	Provides the Linear Regression algorithm and model evaluation metrics.
MLflow  :	Tracks experiments, parameters, metrics and model artifacts.
Joblib  :	Saves and loads the trained machine-learning model.
FastAPI :   Creates an API that accepts house details and returns predicted prices.
Uvicorn :	Runs the FastAPI application as a web server.
Git     :	Tracks changes and maintains versions of the project code.

## 3. Project Structure

```text
house-price-mlops/
├── data/
│   └── house_prices.csv
├── models/
│   └── house_model.joblib
├── generate_data.py
├── train.py
├── api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 4. Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 5. Generate Dataset

```powershell
python generate_data.py
```

## 6. Train the Model

```powershell
python train.py
```

The model is evaluated using MAE, RMSE and R². Experiment details and the model artifact are logged using MLflow. A model copy is saved using Joblib.

## 7. Run the API

```powershell
uvicorn api:app --reload
```

Open the API documentation:

http://127.0.0.1:8000/docs

Use the POST `/predict` endpoint to submit house details and receive a predicted price.

## 8. Model Evaluation

- MAE: 228492.56
- RMSE: 289684.16
- R² Score: 0.976

**Note:** The project uses a synthetically generated dataset for demonstration and learning purposes. These evaluation results do not establish real-world housing-market accuracy.