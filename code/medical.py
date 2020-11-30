## Predicting medical expenses ##

# Importing packages #
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Importing the data #
url = 'https://raw.githubusercontent.com/cinnData/UM-2020-11/main/data/medical.csv'
df = pd.read_csv(url)
df.info()
df.head()

# Target vector #
y = df['charges']

# Numeric features #
X1 = df[['age', 'bmi', 'dependents']]

# Categorical features #
X2 = pd.get_dummies(df[['sex', 'smoker', 'region']])

# Features matrix #
X = pd.concat([X1, X2], axis=1)

# Q1a. Use a linear regression model for predicting the medical cost in terms of the other features
linreg = LinearRegression()
linreg.fit(X, y)

# Q1b. How is the predictive performance of your model?
round(linreg.score(X, y), 3)

# Q2. Plot the actual charges vs the predicted charges. What do you see?
ypred = linreg.predict(X)
from matplotlib import pyplot as plt
plt.figure(figsize=(6,6))
plt.scatter(ypred, y, color='black', s=1)
plt.title('Figure 1. Actual vs predicted charges')
plt.xlabel('Predicted charges')
plt.ylabel('Actual charges')
plt.show()

# Q3. We expect old age, smoking, and obesity tend to be linked to additional health issues, while additional family member dependents may result in an increase in physician visits and preventive care such as vaccinations and yearly physical exams. Is this what you find with your model? How would you modify your equation to cope better with these patterns?
linreg.coef_
X3 = pd.DataFrame({'age2': df['age']**2,
  'age_smoker': (df['smoker'] == 'yes')*df['age'],
  'bmi_smoker': (df['smoker'] == 'yes')*df['bmi']})
X = pd.concat([X, X3], axis=1)
linreg.fit(X, y)
round(linreg.score(X, y), 3)
ypred = linreg.predict(X)
plt.figure(figsize=(6,6))
plt.scatter(ypred, y, color='black', s=1)
plt.title('Figure 1. Actual vs predicted charges')
plt.xlabel('Predicted charges')
plt.ylabel('Actual charges')
plt.show()

# Q4. Do you think that a decision tree model can work here? #
X = pd.concat([X1, X2], axis=1)
treereg = DecisionTreeRegressor(max_depth=4)
treereg.fit(X, y)
plt.figure(figsize=(12,10))
plot_tree(treereg, fontsize=9)
plt.show()
round(treereg.score(X, y), 3)
ypred = treereg.predict(X)
plt.figure(figsize=(6,6))
plt.scatter(ypred, y, color='0.5', s=1)
plt.title('Figure 2. Actual vs predicted charges')
plt.xlabel('Predicted charges')
plt.ylabel('Actual charges')
plt.show()

# Random forest #
rfreg = RandomForestRegressor(max_depth=4, n_estimators=100)
rfreg.fit(X, y)
round(rfreg.score(X, y), 3)
ypred = rfreg.predict(X)
plt.figure(figsize=(6,6))
plt.scatter(ypred, y, color='0.5', s=1)
plt.title('Figure 3. Actual vs predicted charges')
plt.xlabel('Charges')
plt.ylabel('Actual charges')
plt.show()

# Q5. Do your models overfit the data? #
X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.2)
treereg.fit(X_train, y_train)
round(treereg.score(X_train, y_train), 3)
round(treereg.score(X_test, y_test), 3)

# Saving your model to a pkl file #
from joblib import *
dump(treereg, 'treereg.pkl')
newtreereg = load('treereg.pkl')
round(newtreereg.score(X_train, y_train), 3)
