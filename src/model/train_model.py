import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Load data
data_path = r"C:\Users\JT Lappy\Desktop\phase2\voice_fatigue_detection\data\processed\features_labeled.csv"
df = pd.read_csv(data_path)

print("Columns:", df.columns)
print("Labels:", df["label"].unique())

# 🔹 Remove file column
df = df.drop(columns=["file"])

# 🔹 Features & Labels
X = df.drop(columns=["label"])
y = df["label"]

# 🔹 Split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 🔹 Model (Balanced + Stable)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",   # ⭐ FIX (automatic balance)
    random_state=42
)

# 🔹 Train
model.fit(X_train, y_train)

# 🔹 Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 🔹 Threshold tuning
threshold = 0.5   # ⭐ CHANGE THIS (0.4 / 0.5 / 0.6 try)

y_pred = (y_prob > threshold).astype(int)

# 🔹 Results
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
