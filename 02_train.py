import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------
# Step 1: Load dataset with target
# ---------------------------------
df = pd.read_csv(r"C:\DAV_STUDENT_PREDICTION_MODEL\data\dataset_with_target.csv")
print("Dataset loaded:", df.shape)

# ---------------------------------
# Step 2: Drop unnecessary columns
# ---------------------------------
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop(columns=["performance_score", "internal_avg"], errors="ignore")

# Target column
TARGET = "Performance_Category"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------------------------
# Step 3: Identify column types
# ---------------------------------
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ---------------------------------
# Step 4: Preprocessing
# ---------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---------------------------------
# Step 5: Build model pipeline
# ---------------------------------
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ---------------------------------
# Step 6: Split dataset
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------
# Step 7: Train
# ---------------------------------
model.fit(X_train, y_train)
print("Model training complete!")

# ---------------------------------
# Step 8: Evaluate
# ---------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------
# Step 9: Save model
# ---------------------------------
model_path = r"C:\DAV_STUDENT_PREDICTION_MODEL\models\student_model.pkl"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
