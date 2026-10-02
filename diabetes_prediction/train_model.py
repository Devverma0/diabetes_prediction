import pandas as pd, numpy as np, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

df = pd.read_csv("dataset/diabetes.csv")
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results, best_name, best_model, best_f1 = {}, None, None, -1
for name, m in models.items():
    m.fit(X_train_s, y_train)
    p = m.predict(X_test_s)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, p) * 100, 2),
        "precision": round(precision_score(y_test, p) * 100, 2),
        "recall": round(recall_score(y_test, p) * 100, 2),
        "f1": round(f1_score(y_test, p) * 100, 2),
    }
    print(name, results[name])
    if results[name]["f1"] > best_f1:
        best_name, best_model, best_f1 = name, m, results[name]["f1"]

cm = confusion_matrix(y_test, best_model.predict(X_test_s)).tolist()
rf = models["Random Forest"]
importance = {f: round(float(i) * 100, 2)
              for f, i in zip(X.columns, rf.feature_importances_)}
distribution = [int((y == 0).sum()), int((y == 1).sum())]

pickle.dump({
    "model": best_model, "scaler": scaler, "best_name": best_name,
    "results": results, "cm": cm, "importance": importance,
    "distribution": distribution, "features": list(X.columns),
}, open("model.pkl", "wb"))
print("Saved best model:", best_name)