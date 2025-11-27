# auto.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def run_automation(df, target_column, model_name="RandomForest", test_size=0.2):
    """
    Full-featured ML automation:
    - Nulls & duplicates check
    - Categorical encoding
    - Multi-model ML automation
    Returns results dict
    """
    result = {}

    # Step 1: Nulls & Duplicates
    result['nulls'] = df.isnull().sum()
    result['duplicates'] = df.duplicated().sum()
    df_cleaned = df.drop_duplicates()

    # Step 2: Descriptive stats
    result['description'] = df_cleaned.describe(include='all')

    # Step 3: Numeric correlation
    numeric_df = df_cleaned.select_dtypes(include=['float64', 'int64'])
    result['correlation'] = numeric_df.corr() if not numeric_df.empty else pd.DataFrame()

    # Step 4: ML Automation
    if target_column in df_cleaned.columns:
        X = df_cleaned.drop(columns=[target_column])
        y = df_cleaned[target_column]

        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Model selection
        if model_name == "RandomForest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "DecisionTree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result['ml_accuracy'] = accuracy_score(y_test, y_pred)

        # Feature importance for tree-based models
        if model_name in ["RandomForest", "DecisionTree"]:
            fi = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            result['feature_importance'] = fi.sort_values(by='Importance', ascending=False)
        else:
            result['feature_importance'] = None

        # Function for single row prediction
        def predict_single(sample_dict):
            df_sample = pd.DataFrame([sample_dict])
            df_sample = pd.get_dummies(df_sample, drop_first=True)
            # Align columns
            df_sample = df_sample.reindex(columns=X.columns, fill_value=0)
            return model.predict(df_sample)[0]
        result['predict_single'] = predict_single
    else:
        result['ml_accuracy'] = None
        result['feature_importance'] = None
        result['predict_single'] = None

    result['cleaned_df'] = df_cleaned
    return result
