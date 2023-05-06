import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import xgboost as xgb


def load_and_preprocess_data():
    data = pd.read_csv('variant3.csv')

    data.drop(['time_taken', 'twitter_hastags', 'Num_multiplex'], axis=1, inplace=True)

    data['Total_Rating'] = data[
        ['Lead_Actor_Rating', 'Lead_Actress_rating', 'Director_rating', 'Producer_rating', 'Critic_rating']].sum(axis=1)
    data['Avg_Rating'] = data[
        ['Lead_Actor_Rating', 'Lead_Actress_rating', 'Director_rating', 'Producer_rating', 'Critic_rating']].mean(
        axis=1)
    data['Budget_per_Second'] = data['Budget'] / data['Length']
    data['Trailer_views_per_Day'] = data['Trailer_views'] / data['Days']

    return data


def prepare_classification_data(data):
    bins = [0, 2.5, 5]
    labels = ['Low', 'High']
    data['Critic_rating'] = pd.cut(data['Critic_rating'], bins=bins, labels=labels)
    data['Critic_rating'] = LabelEncoder().fit_transform(data['Critic_rating'])
    return data


def prepare_feature_matrix_and_target(data):
    X = data.drop('Critic_rating', axis=1)
    y = data['Critic_rating']
    return X, y


def split_and_standardize_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def regression_methods(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(),
        'Random Forest Regression': RandomForestRegressor(),
        'XGBoost Regression': xgb.XGBRegressor()
    }

    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if r2 > best_r2:
            best_model = name
            best_r2 = r2

        print(f"{name} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}")

    print(f"Best Regression Model: {best_model}, R-squared: {best_r2:.2f}")


def classification_methods(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine Classifier': SVC(),
        'Random Forest Classifier': RandomForestClassifier(),
        'XGBoost Classifier': xgb.XGBClassifier()
    }

    best_model = None
    best_accuracy = -np.inf

    for name, model in models.items():
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)

    if acc > best_accuracy:
        best_model = name
        best_accuracy = acc

    print(f"{name} - Accuracy: {acc:.2f}")
    print(f"Classification Report:\n{cls_report}")

print(f"Best Classification Model: {best_model}, Accuracy: {best_accuracy:.2f}")
def predict_critic_rating(method):
data = load_and_preprocess_data()
    if method == "classification":
        data = prepare_classification_data(data)

    X, y = prepare_feature_matrix_and_target(data)
    X_train, X_test, y_train, y_test = split_and_standardize_data(X, y)

    if method == "regression":
        regression_methods(X_train, X_test, y_train, y_test)
    elif method == "classification":
        classification_methods(X_train, X_test, y_train, y_test)
    else:
        raise ValueError("Invalid method. Choose 'regression' or 'classification'")
if name == "main":
print("Regression Results:")
predict_critic_rating("regression")
print("\nClassification Results:")
predict_critic_rating("classification")
