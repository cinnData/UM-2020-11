## Modeling loan acceptance ##

# Importing packages #
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# Importing the data #
url = 'https://raw.githubusercontent.com/cinnData/UM-2020-11/main/data/loan.csv'
df = pd.read_csv(url)
df.info()
df.head()

# Rate of success #
success = df['loan'].mean()
round(success, 3)

# Checking the data #
(df['mortgage'] == 0).mean()
df['income'].describe()
df['education'].value_counts()
df['family'].value_counts()

# Target vector #
y = df['loan']

# Numeric features #
X1 = df[['age', 'income', 'ccavg', 'mortgage', 'secacc',
  'cdacc', 'online', 'ccard']]

# Categorical features #
X2 = pd.get_dummies(df['family'])
X2.columns = ['fam1', 'fam2', 'fam3', 'fam4']
X3 = pd.get_dummies(df['education'])
X3.columns = ['edu1', 'edu2', 'edu3']

# Features matrix #
X = pd.concat([X1, X2, X3], axis=1)
X.head()

# Q1. Logistic regression equation #
logclf = LogisticRegression(solver='liblinear')
logclf.fit(X, y)
round(logclf.score(X, y), 3)

# Q2. Distribution of predictive scores #
scores = logclf.predict_proba(X)[:, 1]
from matplotlib import pyplot as plt
fig, (fig1, fig2) = plt.subplots(1, 2, figsize = (14,6))
fig1.set_title('Figure 1a. Scores (acceptants)')
fig1.hist(scores[y == 1], color='gray', rwidth=0.95, bins=20)
fig1.set_xlabel('Acceptance score')
fig2.set_title('Figure 1b. Scores (non-acceptants)')
fig2.hist(scores[y == 0], color='gray', rwidth=0.95, bins=20)
fig2.set_xlabel('Acceptance score')
plt.show()

# Q3a. Cutoff 0.5 #
y_pred = logclf.predict(X)
pd.crosstab(y, y_pred)

# Q3b. Cutoff 0.1 #
y_pred = (scores > 0.1).astype('int')
pd.crosstab(y, y_pred)
tp = sum((y == 1) & (y_pred == 1))/sum(y == 1)
fp = sum((y == 0) & (y_pred == 1))/sum(y == 0)
round(tp, 3), round(fp, 3)

# Q4. Additional customers #
48*1000*success*tp - 2*1000*(1-success)*fp

# Q5. Optimal cutoff #
def profit(cutoff):
    y_pred = (scores > cutoff).astype('int')
    pd.crosstab(y, y_pred)
    tp = sum((y == 1) & (y_pred == 1))/sum(y == 1)
    fp = sum((y == 0) & (y_pred == 1))/sum(y == 0)
    profit = 48*1000*success*tp - 2*1000*(1-success)*fp
    return profit
