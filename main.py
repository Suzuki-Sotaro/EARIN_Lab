import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA


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

    # Remove unnecessary features
    # Uncomment the line below if there are any unnecessary features to be removed
    # data = data.drop(columns=['Unnecessary_feature_1', 'Unnecessary_feature_2'])

    # Generate new features from existing features
    genre_success_rate = data.groupby('Genre')['Critic_rating'].mean().to_dict()
    data['Genre_Success_Rate'] = data['Genre'].map(genre_success_rate)

    return data


def feature_selection(data, method='correlation', n_features=None):
    X = data.drop(columns=['Critic_rating'])
    y = data['Critic_rating']

    if method == 'correlation':
        # Filter method: Correlation
        corr_matrix = data.corr()
        high_corr_features = corr_matrix['Critic_rating'].abs().sort_values(ascending=False).head(n_features).index
        X = X[high_corr_features]

    elif method == 'rfe':
        # Wrapper method: Recursive Feature Elimination
        estimator = LinearRegression() if data['Critic_rating'].dtype == 'float64' else LogisticRegression()
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        X = X[X.columns[rfe.support_]]

    elif method == 'embedding':
        # Embedded method: SelectFromModel with Lasso regularization
        estimator = Lasso(alpha=0.01) if data['Critic_rating'].dtype == 'float64' else LogisticRegression(penalty='l1',
                                                                                                          solver='liblinear',
                                                                                                          C=0.1)
        sfm = SelectFromModel(estimator)
        sfm.fit(X, y)
        X = X[X.columns[sfm.get_support()]]

    elif method == 'filter':
        # Filter method: SelectKBest with mutual information
        kbest = SelectKBest(mutual_info_regression if data['Critic_rating'].dtype == 'float64' else mutual_info_classif,
                            k=n_features)
        kbest.fit(X, y)
        X = X[X.columns[kbest.get_support()]]

    return X


def feature_extraction(X, n_components=None):
    # Dimensionality Reduction: PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca


def prepare_classification_data(data):
    bins = [0, 7, 8, 10]
    labels = [0, 1, 2]
    data['Critic_rating'] =
    pd.cut(data['Critic_rating'], bins=bins, labels=labels)
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
    # Initialize models with proper hyperparameters
    models = {
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(kernel='linear', C=1, epsilon=0.1),
        "Random Forest Regression": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost Regression": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
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
    # Initialize models with proper hyperparameters
    models = {
        "Logistic Regression": LogisticRegression(C=1, random_state=42),
        "Support Vector Machine Classifier": SVC(kernel='linear', C=1, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
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

    # Feature selection
    X_selected = feature_selection(data, method='correlation', n_features=5)

    # Feature extraction
    X_pca = feature_extraction(X, n_components=5)

    X_train, X_test, y_train, y_test = split_and_standardize_data(X_selected, y)


X_pca_train, X_pca_test, y_pca_train, y_pca_test = split_and_standardize_data(X_pca, y)
print("Using selected features:")
if method == 'regression':
    regression_methods(X_train, X_test, y_train, y_test)
elif method == 'classification':
    classification_methods(X_train, X_test, y_train, y_test)
else:
    print("Invalid method. Choose 'regression' or 'classification'.")

print("\nUsing PCA for feature extraction:")
if method == 'regression':
    regression_methods(X_pca_train, X_pca_test, y_pca_train, y_pca_test)
elif method == 'classification':
    classification_methods(X_pca_train, X_pca_test, y_pca_train, y_pca_test)
else:
    print("Invalid method. Choose 'regression' or 'classification'.")
