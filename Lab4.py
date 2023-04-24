import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data():
    # Load the dataset
    data = pd.read_csv('variant3.csv')

    # Preprocess the data
    data['3D_available'] = data['3D_available'].apply(lambda x: 1 if x == 'YES' else 0)
    data['Start_Tech_Oscar'] = data['Start_Tech_Oscar'].apply(lambda x: 1 if x == 'YES' else 0)
    label_encoder = LabelEncoder()
    data['Genre'] = label_encoder.fit_transform(data['Genre'])

    # Handle missing values
    data['Time_taken'].fillna(data['Time_taken'].mean(), inplace=True)

    return data


def prepare_classification_data(data):
    bins = [0, 7, 8, 10]
    labels = [0, 1, 2]
    data['Critic_rating'] = pd.cut(data['Critic_rating'], bins=bins, labels=labels)
    return data


def prepare_feature_matrix_and_target(data):
    X = data.drop(columns=['Critic_rating'])
    y = data['Critic_rating']
    return X, y


def split_and_standardize_data(X, y):
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def regression_methods(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel='linear'),
        "Random Forest Regression": RandomForestRegressor(),
        "XGBoost Regression": XGBRegressor()
    }

    r2_scores = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_scores[name] = r2
        print(f"{name} - Mean Squared Error: {mse}, R-squared: {r2}")

    # Find the best regression method
    best_method = max(r2_scores, key=r2_scores.get)
    print(f"The best regression method is: {best_method}")


def classification_methods(X_train, X_test, y_train, y_test):
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine Classifier": SVC(kernel='linear'),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBoost Classifier": XGBClassifier()
    }

    accuracies = {}

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        accuracies[name] = accuracy
        print(f"{name} - Accuracy: {accuracy}\nClassification Report:\n{report}")

    # Find the best classification method
    best_method = max(accuracies, key=accuracies.get)
    print(f"The best classification method is: {best_method}")


def predict_critic_rating(method='regression'):
    data = load_and_preprocess_data()

    if method == 'classification':
        data = prepare_classification_data(data)

    X, y = prepare_feature_matrix_and_target(data)
    X_train, X_test, y_train, y_test = split_and_standardize_data(X, y)

    if method == 'regression':
        regression_methods(X_train, X_test, y_train, y_test)

    elif method == 'classification':
        classification_methods(X_train, X_test, y_train, y_test)

    else:
        print("Invalid method. Choose 'regression' or 'classification'.")


# Example usage:
predict_critic_rating('regression')  # For regression methods
predict_critic_rating('classification')  # For classification methods
