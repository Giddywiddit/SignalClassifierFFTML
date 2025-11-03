import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib, os

def train():
    X = np.load("data/features.npy")
    y = np.load("data/labels.npy")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=300)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"scaler": scaler, "model": clf}, "models/fft_clf.joblib")

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(y_te, y_pred, digits=4))

if __name__ == "__main__":
    train()
