import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import numpy as np

def automate_training(df, target):
    # Separate X and y
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical columns
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Encode target if needed
    target_is_cat = False
    if y.dtype == "object":
        target_is_cat = True
        y_le = LabelEncoder()
        y = y_le.fit_transform(y.astype(str))
        encoders[target] = y_le

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Auto choose model
    if target_is_cat or len(y.unique()) <= 20:
        model = RandomForestClassifier()
        model_type = "classification"
    else:
        model = RandomForestRegressor()
        model_type = "regression"

    # Train
    model.fit(X_train, y_train)

    # Score
    preds = model.predict(X_test)

    if model_type == "classification":
        score = accuracy_score(y_test, preds)
    else:
        score = np.sqrt(mean_squared_error(y_test, preds))

    # Save model + encoders
    joblib.dump(model, "model.pkl")
    joblib.dump(encoders, "encoders.pkl")

    return model_type, score, list(X.columns)
