
import pandas as pd
import numpy as np

# Generate 200 sample house records
np.random.seed(42)

data = {
    "area_sqft": np.random.randint(800, 3001, 200),
    "bedrooms": np.random.randint(1, 6, 200),
    "bathrooms": np.random.randint(1, 5, 200),
    "house_age": np.random.randint(0, 31, 200)
}

# Calculate sample house prices
price = (
    500000
    + data["area_sqft"] * 2500
    + data["bedrooms"] * 500000
    + data["bathrooms"] * 200000
    - data["house_age"] * 50000
    + np.random.normal(0, 300000, 200)
)

data["price"] = price.round(0).astype(int)

# Convert the data into a Pandas DataFrame
df = pd.DataFrame(data)

# Save the dataset as a CSV file
df.to_csv("data/house_prices.csv", index=False)

print("Dataset created successfully!")
print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())
