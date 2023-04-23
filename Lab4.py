import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('variant3.csv')

# Preprocess the data
data['3D_available'] = data['3D_available'].apply(lambda x: 1 if x == 'YES' else 0)
data['Start_Tech_Oscar'] = data['Start_Tech_Oscar'].apply(lambda x: 1 if x == 'YES' else 0)
label_encoder = LabelEncoder()
data['Genre'] = label_encoder.fit_transform(data['Genre'])

# Handle missing values
data['Time_taken'].fillna(data['Time_taken'].mean(), inplace=True)

# Define the feature matrix and target variable
X = data.drop(columns=['Critic_rating'])
y = data['Critic_rating']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Method 1: Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred_lr = linear_regression.predict(X_test)

# Method 2: Support Vector Machine Regression
svm_regression = SVR(kernel='linear')
svm_regression.fit(X_train, y_train)
y_pred_svr = svm_regression.predict(X_test)

# Evaluate the performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_lr = r2_score(y_test, y_pred_lr)
r2_svr = r2_score(y_test, y_pred_svr)

print(f"Linear Regression - Mean Squared Error: {mse_lr}, R-squared: {r2_lr}")
print(f"Support Vector Regression - Mean Squared Error: {mse_svr}, R-squared: {r2_svr}")

if r2_lr > r2_svr:
    print("Linear Regression is the better method.")
else:
    print("Support Vector Regression is the better method.")
