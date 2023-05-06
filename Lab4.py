import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


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

    # Drop unnecessary columns
    data.drop(columns=['Twitter_hastags', 'Num_multiplex'], inplace=True)

    # Add new features
    data['Total_Rating'] = data[['Lead_ Actor_Rating', 'Lead_Actress_rating', 'Director_rating', 'Producer_rating', 'Critic_rating']].sum(axis=1)
    data['Avg_Rating'] = data[['Lead_ Actor_Rating', 'Lead_Actress_rating', 'Director_rating', 'Producer_rating', 'Critic_rating']].mean(axis=1)
    data['Budget_per_Second'] = data['Budget'] / (data['Movie_length'] * 60)
    data['Trailer_views_per_Day'] = data['Trailer_views'] / data['Time_taken']
    
    # Check for infinity and NaN values
    inf_values = data.applymap(np.isinf)
    nan_values = data.isnull()
    problem_values = inf_values | nan_values
    
    # Print rows and columns with problematic values
    print("Rows and columns with inf, -inf, or NaN values:")
    for index, row in problem_values.iterrows():
        for col, value in row.items():
            if value:
                print(f"Index: {index}, Column: {col}")
    
    # Check for infinity and replace with NaN if found
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill any NaN values with the mean of the column
    data.fillna(data.mean(), inplace=True)

    return data


def select_features(X, y, k=10):
    # Select k best features based on mutual information
    selector = SelectKBest(mutual_info_regression, k=k)
    X_new = selector.fit_transform(X, y)

    # Get the names of the selected features
    selected_features = X.columns[selector.get_support()]

    return X_new, selected_features


def extract_features(X, n_components=2):
    # Apply PCA to extract new features
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)

    return X_new

def prepare_classification_data(data):
    bins = [0, 7, 8, 10]
    labels = [0, 1, 2]
    data['Critic_rating'] = pd.cut(data['Critic_rating'], bins=bins, labels=labels)
    return data



def prepare_feature_matrix_and_target(data, feature_selection=True, feature_extraction=True):
    X = data.drop(columns=['Critic_rating'])
    y = data['Critic_rating']

    if feature_selection:
        X, selected_features = select_features(X, y)
        print("Selected features:", selected_features)

    if feature_extraction:
        X = extract_features(X)
    
    return X, y


def split_and_standardize_data(X, y):
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    if X_train.ndim > 1:
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

    # Define parameter grids for each model
    param_grids = {
        "Linear Regression": {},
        "Support Vector Regression": {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.5, 1]},
        "Random Forest Regression": {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
        "XGBoost Regression": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1, 0.3]}
    }
    r2_scores = {}

    # Train and evaluate each model with GridSearchCV
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_scores[name] = r2
        print(f"{name} - Best Parameters: {grid_search.best_params_}")
        print(f"Mean Squared Error: {mse}, R-squared: {r2}")

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
    param_grids = {
        "Logistic Regression": {'C': [0.1, 1, 10]},
        "Support Vector Machine Classifier": {'C': [0.1, 1, 10]},
        "Random Forest Classifier": {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]},
        "XGBoost Classifier": {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1, 0.3]}
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



def predict_critic_rating(method='regression', feature_selection=True, feature_extraction=True):
    data = load_and_preprocess_data()

    if method == 'classification':
        data = prepare_classification_data(data)

    X, y = prepare_feature_matrix_and_target(data, feature_selection=feature_selection, feature_extraction=feature_extraction)
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



def predict_critic_rating(method='regression'):
    data = load_and_preprocess_data()

