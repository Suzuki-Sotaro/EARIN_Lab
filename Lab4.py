import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

def predict_critic_rating(method='regression'):
    # Load the dataset
    data = pd.read_csv('variant3.csv')

    # Preprocess the data
    data['3D_available'] = data['3D_available'].apply(lambda x: 1 if x == 'YES' else 0)
    data['Start_Tech_Oscar'] = data['Start_Tech_Oscar'].apply(lambda x: 1 if x == 'YES' else 0)
    label_encoder = LabelEncoder()
    data['Genre'] = label_encoder.fit_transform(data['Genre'])

    # Handle missing values
    data['Time_taken'].fillna(data['Time_taken'].mean(), inplace=True)

    if method == 'classification':
        # Convert Critic_rating into discrete categories
        # Convert Critic_rating into discrete categories
        bins = [0, 7, 8, 10]
        labels = [0, 1, 2]
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

    if method == 'regression':
        # Regression methods

        # Method 1: Linear Regression
        linear_regression = LinearRegression()
        linear_regression.fit(X_train, y_train)
        y_pred_lr = linear_regression.predict(X_test)

        # Method 2: Support Vector Machine Regression
        svm_regression = SVR(kernel='linear')
        svm_regression.fit(X_train, y_train)
        y_pred_svr = svm_regression.predict(X_test)
        
        # Method 3: Random Forest Regressor
        random_forest_regressor = RandomForestRegressor()
        random_forest_regressor.fit(X_train, y_train)
        y_pred_rf = random_forest_regressor.predict(X_test)

        # Method 4: XGBoost Regressor
        xgb_regressor = XGBRegressor()
        xgb_regressor.fit(X_train, y_train)
        y_pred_xgb = xgb_regressor.predict(X_test)

        # Evaluate the performance
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_svr = mean_squared_error(y_test, y_pred_svr)
        r2_lr = r2_score(y_test, y_pred_lr)
        r2_svr = r2_score(y_test, y_pred_svr)
        
        # Evaluate the performance
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        r2_rf = r2_score(y_test, y_pred_rf)
        r2_xgb = r2_score(y_test, y_pred_xgb)

        print(f"Linear Regression - Mean Squared Error: {mse_lr}, R-squared: {r2_lr}")
        print(f"Support Vector Regression - Mean Squared Error: {mse_svr}, R-squared: {r2_svr}")
        print(f"Random Forest Regression - Mean Squared Error: {mse_rf}, R-squared: {r2_rf}")
        print(f"XGBoost Regression - Mean Squared Error: {mse_xgb}, R-squared: {r2_xgb}")

        # Find the best regression method
        methods = ['Linear Regression', 'Support Vector Regression', 'Random Forest Regression', 'XGBoost Regression']
        r2_scores = [r2_lr, r2_svr, r2_rf, r2_xgb]
        best_method_index = np.argmax(r2_scores)
        print(f"The best regression method is: {methods[best_method_index]}")

            
    elif method == 'classification':
        # Classification methods

        # Method 1: Logistic Regression
        logistic_regression = LogisticRegression()
        logistic_regression.fit(X_train, y_train)
        y_pred_lr = logistic_regression.predict(X_test)

        # Method 2: Support Vector Machine Classification
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        y_pred_svm = svm_classifier.predict(X_test)
        
        # Method 3: Random Forest Classifier
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(X_train, y_train)
        y_pred_rf = random_forest_classifier.predict(X_test)
 
        # Method 4: XGBoost Classifier
        xgb_classifier = XGBClassifier()
        xgb_classifier.fit(X_train, y_train)
        y_pred_xgb = xgb_classifier.predict(X_test)

        # Evaluate the performance
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        report_lr = classification_report(y_test, y_pred_lr)
        report_svm = classification_report(y_test, y_pred_svm)
        
        # Evaluate the performance
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        report_rf = classification_report(y_test, y_pred_rf)
        report_xgb = classification_report(y_test, y_pred_xgb)

        print(f"Logistic Regression - Accuracy: {accuracy_lr}\nClassification Report:\n{report_lr}")
        print(f"Support Vector Machine Classifier - Accuracy: {accuracy_svm}\nClassification Report:\n{report_svm}")
        print(f"Random Forest Classifier - Accuracy: {accuracy_rf}\nClassification Report:\n{report_rf}")
        print(f"XGBoost Classifier - Accuracy: {accuracy_xgb}\nClassification Report:\n{report_xgb}")

        # Find the best classification method
        methods = ['Logistic Regression', 'Support Vector Machine Classifier', 'Random Forest Classifier', 'XGBoost Classifier']
        accuracies = [accuracy_lr, accuracy_svm, accuracy_rf, accuracy_xgb]
        best_method_index = np.argmax(accuracies)
        print(f"The best classification method is: {methods[best_method_index]}")
        
        if accuracy_lr > accuracy_svm:
            print("Logistic Regression is the better method.")
        else:
            print("Support Vector Machine Classifier is the better method.")
    else:
        print("Invalid method. Choose 'regression' or 'classification'.")


# Example usage:
predict_critic_rating('regression')  # For regression methods
predict_critic_rating('classification')  # For classification methods

