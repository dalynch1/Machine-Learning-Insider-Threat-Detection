import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# -----------------------------
# Load dataset
# -----------------------------

# change path if needed
df = pd.read_csv("insider_threat_clean_dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())
print(df.info())


# -----------------------------
# Separate features and target
# -----------------------------

# target variable
y = df["is_malicious"]

# input features
X = df.drop("is_malicious", axis=1)


# -----------------------------
# Identify categorical columns
# -----------------------------

categorical_columns = [
    "employee_department",
    "employee_campus",
    "employee_position",
    "employee_origin_country"
]


# -----------------------------
# Convert text columns to numbers
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
    remainder="passthrough"
)


# -----------------------------
# Create Random Forest model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_depth=10,       # limit tree depth
    random_state=1,
    class_weight='balanced'
    
)


# -----------------------------
# Build pipeline
# -----------------------------

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])


# -----------------------------
# Split data into train/test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)


# -----------------------------
# Train the model
# -----------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)


# -----------------------------
# Make predictions
# -----------------------------

y_pred = pipeline.predict(X_test)


# -----------------------------
# Model evaluation
# -----------------------------

print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# -----------------------------
# ROC Curve + AUC Score
# -----------------------------

# get probability predictions
y_probs = pipeline.predict_proba(X_test)[:,1]

# calculate ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# calculate AUC score
roc_auc = auc(fpr, tpr)

print("\nAUC Score:", roc_auc)


# plot ROC curve
plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="ROC Curve (AUC = %0.3f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.tight_layout()
plt.legend()

plt.show()


# -----------------------------
# Feature Importance
# -----------------------------

# get feature names after encoding
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

# get importance scores
importances = pipeline.named_steps["model"].feature_importances_

# create dataframe
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# sort values
feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))


# -----------------------------
# Plot Feature Importance
# -----------------------------

top_features = feature_importance.head(10)

plt.figure(figsize=(8,5))

sns.barplot(
    data=top_features,
    x="importance",
    y="feature"
)

plt.title("Top Features Influencing Malicious Behavior")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()