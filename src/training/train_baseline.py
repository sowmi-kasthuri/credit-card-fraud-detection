import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

import os
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:///D:/Sowmi/ai-learning-journey/credit-card-fraud-detection/mlruns")

def load_data():
    df = pd.read_csv("data/raw/creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X,y

def split_data(X,y):
    return train_test_split(
        X,y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled, scaler

def train_logistic_regression(X_train,y_train):
    model = LogisticRegression(max_iter=500)
    model.fit(X_train,y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("ROC-AUC:", auc)

    return precision, recall, f1, auc

def main():

    with mlflow.start_run():

        # 1. Load data
        X,y = load_data()

        # 2. Train-test split
        X_train,X_test,y_train,y_test = split_data(X,y)

        # 3. Scale
        X_train_scaled,X_test_scaled,scaler = scale_data(X_train,X_test)

        # 4. Train model
        model = train_logistic_regression(X_train_scaled,y_train)

        # 5. Evaluate
        precision, recall, f1, auc = evaluate_model(model, X_test_scaled, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 500)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(model, "model")



if __name__ == "__main__":
    main()