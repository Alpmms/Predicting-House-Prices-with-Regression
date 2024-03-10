#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset (replace with your actual dataset)
data = pd.read_csv('house_prices.csv')

# Preprocessing: handle missing values, encode categorical features, etc.

# Split data into features (X) and target (y)
X = data[['SquareFootage', 'Bedrooms', 'Location']]
y = data['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R-squared: {r2:.2f}")

# Save the model for future use
# ...

# Visualize predictions vs. actual prices
# ...



# In[ ]:





# In[ ]:




