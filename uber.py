#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
data = pd.read_csv("/content/uber.csv")
df = pd.DataFrame(data)
df.head()
df.info()
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df.describe()

#Remove Null Values
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

#Remove outliners
plt.boxplot(df['fare_amount'])
#Remove Outliers
q_low = df["fare_amount"].quantile(0.01)
q_hi  = df["fare_amount"].quantile(0.99)

df = df[(df["fare_amount"] < q_hi) & (df["fare_amount"] > q_low)]
plt.boxplot(df['fare_amount'])

#corr matrix heatmap
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)
plt.figure(figsize=(10, 8))  # adjust figure size
sns.heatmap(
    corr_matrix,
    annot=True,        # show correlation values
    fmt=".2f",         # format to 2 decimal places
    cmap="coolwarm",   # color palette (other options: 'viridis', 'magma', 'crest')
    linewidths=0.5     # adds space between cells
)
plt.title("Correlation Matrix Heatmap")
plt.show()


#Time to apply learning models
from sklearn.model_selection import train_test_split
#Take x as predictor variable
x = df.drop("fare_amount", axis = 1)
#And y as target variable
y = df['fare_amount']
#Necessary to apply model
x['pickup_datetime'] = pd.to_numeric(pd.to_datetime(x['pickup_datetime']))
x = x.loc[:, x.columns.str.contains('^Unnamed')]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
from sklearn.linear_model import LinearRegression

lrmodel = LinearRegression()
lrmodel.fit(x_train, y_train)

predict = lrmodel.predict(x_test)
# evaluation

from sklearn.metrics import mean_squared_error, r2_score

lr_rmse = np.sqrt(mean_squared_error(y_test, predict))
lr_r2 = r2_score(y_test, predict)

print("Linear Regression → RMSE:", lr_rmse, "R²:", lr_r2)

#Random forest implementation
#Let's Apply Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfrmodel = RandomForestRegressor(n_estimators = 100, random_state = 101)
#Fit the Forest
rfrmodel.fit(x_train, y_train)
rfrmodel_pred = rfrmodel.predict(x_test)
rfr_rmse = np.sqrt(mean_squared_error(y_test, rfrmodel_pred))
rfr_r2 = r2_score(y_test, rfrmodel_pred)

print("Random Forest → RMSE:", rfr_rmse, "R²:", rfr_r2)
