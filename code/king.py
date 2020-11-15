## House sales in King County ##

# Loading packages #
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Importing the data #
url = 'https://raw.githubusercontent.com/cinnData/UM-2020-11/main/data/king.csv'
df = pd.read_csv(url)
df.info()
df.head()

# Rescaling #
df['price'] = df['price']/1000

# Q1. Distribution of sale price #
plt.figure(figsize=(8,6))
plt.title('Figure 1. Sale price')
plt.hist(df['price'], color='gray', rwidth=0.97)
plt.xlabel('Price (thousands)')
plt.show()

# Q2a. Linear regression #
y = df['price']
X = df[['lat', 'long', 'bedrooms', 'bathrooms', 'sqft_above',
  'sqft_basement', 'sqft_lot', 'floors', 'waterfront',
  'condition', 'yr_built', 'yr_renovated']]
linreg = LinearRegression()
linreg.fit(X, y)

# Q2b. Evaluating the regression model #
round(linreg.score(X, y), 3)
df['pred1'] = linreg.predict(X)
r1 = df[['price', 'pred1']].corr().iloc[0, 1]
round(r1**2, 3)

# Q3. Plot actual price vs predicted price #
plt.figure(figsize=(6,6))
plt.scatter(df['pred1'], df['price'], color='black', s=1)
plt.title('Figure 2. Actual vs predicted price')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)')
plt.show()

# Q4a. Decision tree regressor #
treereg = DecisionTreeRegressor(max_leaf_nodes=16)
treereg.fit(X, y)
round(treereg.score(X, y), 3)
plt.figure(figsize=(6,6))
plt.scatter(treereg.predict(X), y, color='black', s=1)
plt.title('Figure 3. Actual vs predicted price (decision tree)')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)')
plt.show()

# Q4b. Random forest regressor #
rfreg = RandomForestRegressor(max_leaf_nodes=16, n_estimators=100)
rfreg.fit(X, y)
round(rfreg.score(X, y), 3)
plt.figure(figsize=(6,6))
plt.scatter(rfreg.predict(X), y, color='black', s=1)
plt.title('Figure 4. Actual vs predicted price (random forest)')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)')
plt.show()

# Q4c. Gradient boosted regressor (XGBoost version) #
xgbreg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
xgbreg.fit(X, y)
round(xgbreg.score(X, y), 3)
plt.figure(figsize=(6,6))
plt.scatter(xgbreg.predict(X), y, color='black', s=1)
plt.title('Figure 5. Actual vs predicted price (xgboost)')
plt.xlabel('Predicted price (thousands)')
plt.ylabel('Actual price (thousands)')
plt.show()
