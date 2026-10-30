

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("data/house_prices.csv")

# Step 2: Select independent variables (X)
X = df[["area_sqft", "bedrooms", "bathrooms", "house_age"]]

# Step 3: Select dependent variable (y)
y = df["price"]

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create the model
model = LinearRegression()

# Step 6: Set MLflow experiment
mlflow.set_experiment("House Price Prediction")

# Step 7: Start an MLflow run
with mlflow.start_run():

    # Train the model
    model.fit(X_train, y_train)

    joblib.dump(model, "models/house_model.joblib")

    print("Model saved successfully using Joblib!")

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Record model parameters
    mlflow.log_param("model", "Linear Regression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Record evaluation metrics
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Save the trained model in MLflow
    mlflow.sklearn.log_model(
        model,
        name="house_price_model",
        input_example=X_train.head(5)
    )

    print("Model training completed!")
    print("\nModel Evaluation:")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    print("\nMLflow tracking completed!")
