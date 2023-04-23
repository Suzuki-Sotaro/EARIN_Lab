import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
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

# Convert Critic_rating into discrete categories
bins = [0, 3, 6, 10]
labels = ['Low', 'Medium', 'High']
data['Critic_rating'] = pd.cut(data['Critic_rating'], bins=bins, labels=labels)

# Define the feature matrix and target variable
X = data.drop(columns=['Critic_rating'])
y = data['Critic_rating']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Method 1: Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_lr = logistic_regression.predict(X_test)

# Method 2: Support Vector Machine Classification
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the performance
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_lr = classification_report(y_test, y_pred_lr)
report_svm = classification_report(y_test, y_pred_svm)

print(f"Logistic Regression - Accuracy: {accuracy_lr}\nClassification Report:\n{report_lr}")
print(f"Support Vector Machine Classifier - Accuracy: {accuracy_svm}\nClassification Report:\n{report_svm}")

if accuracy_lr > accuracy_svm:
    print("Logistic Regression is the better method.")
else:
    print("Support Vector Machine Classifier is the better method.")
