import pandas as pd
import numpy as np

print("=" * 70)
print("        TITANIC SURVIVAL PREDICTION USING ML")
print("=" * 70)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("data/train.csv")

print("\n✅ Dataset Loaded Successfully!")

# ==========================================================
# FIRST 5 ROWS
# ==========================================================

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# ==========================================================
# LAST 5 ROWS
# ==========================================================

print("\n========== LAST 5 ROWS ==========\n")
print(df.tail())

# ==========================================================
# DATASET SHAPE
# ==========================================================

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

# ==========================================================
# COLUMN NAMES
# ==========================================================

print("\n========== COLUMN NAMES ==========\n")
print(df.columns.tolist())

# ==========================================================
# DATA TYPES
# ==========================================================

print("\n========== DATA TYPES ==========\n")
print(df.dtypes)

# ==========================================================
# DATASET INFORMATION
# ==========================================================

print("\n========== DATASET INFORMATION ==========\n")
df.info()

# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ==========================================================
# DUPLICATE VALUES
# ==========================================================

print("\n========== DUPLICATE VALUES ==========\n")
print("Duplicate Rows :", df.duplicated().sum())

# ==========================================================
# STATISTICAL SUMMARY
# ==========================================================

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

# ==========================================================
# UNIQUE VALUES
# ==========================================================

print("\n========== UNIQUE VALUES ==========\n")

for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}")
    print(df[col].value_counts())

print("\n✅ PART 1 COMPLETED SUCCESSFULLY!")

# ==========================================================
# PART 2 : DATA CLEANING & VISUALIZATION
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 70)
print("              DATA CLEANING")
print("=" * 70)

# Missing Values Before Cleaning
print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

# Fill Missing Values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin Column (Too Many Missing Values)
if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)

# Remove Duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows : {duplicates}")

df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# ==========================================================
# SURVIVAL COUNT GRAPH
# ==========================================================

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.tight_layout()

plt.savefig("images/survival_count.png")
plt.close()

print("✓ Survival Count Graph Saved")

# ==========================================================
# GENDER VS SURVIVAL
# ==========================================================

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Gender vs Survival")
plt.tight_layout()

plt.savefig("images/gender_survival.png")
plt.close()

print("✓ Gender Survival Graph Saved")

# ==========================================================
# AGE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(7,5))
sns.histplot(df["Age"], bins=25, kde=True)

plt.title("Age Distribution")
plt.tight_layout()

plt.savefig("images/age_distribution.png")
plt.close()

print("✓ Age Distribution Graph Saved")

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(10,8))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(
    numeric_df.corr(),
    annot=False,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("images/correlation_heatmap.png")
plt.close()

print("✓ Correlation Heatmap Saved")

print("\n✅ PART 2 COMPLETED SUCCESSFULLY!")

# ==========================================================
# PART 3 : FEATURE ENGINEERING
# ==========================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("\n" + "=" * 70)
print("      FEATURE ENGINEERING & TRAIN TEST SPLIT")
print("=" * 70)

# Drop unnecessary columns
drop_columns = ["PassengerId", "Name", "Ticket"]

for col in drop_columns:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Label Encoding
encoder = LabelEncoder()

categorical_columns = ["Sex", "Embarked"]

for col in categorical_columns:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])

print("\n✓ Label Encoding Completed Successfully!")

# Features & Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\n========== FEATURES ==========")
print("Features Shape :", X.shape)

print("\n========== TARGET ==========")
print("Target Shape :", y.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

print("\n✅ PART 3 COMPLETED SUCCESSFULLY!")

# ==========================================================
# PART 4 : MACHINE LEARNING MODEL
# ==========================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import joblib

print("\n" + "=" * 70)
print("              MACHINE LEARNING MODEL")
print("=" * 70)

# Create Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

print("\n✓ Model Training Completed Successfully!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========\n")

print(f"Accuracy : {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()

print("\n✓ Confusion Matrix Saved")

# Feature Importance
importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,6))

sns.barplot(
    data=feature_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("images/feature_importance.png")
plt.close()

print("✓ Feature Importance Graph Saved")

# Save Model
joblib.dump(model, "model/titanic_model.pkl")

print("\n✓ Model Saved Successfully!")

print("\n✅ PART 4 COMPLETED SUCCESSFULLY!")

# ==========================================================
# PART 5 : TITANIC SURVIVAL PREDICTION
# ==========================================================

print("\n" + "=" * 70)
print("         TITANIC SURVIVAL PREDICTION SYSTEM")
print("=" * 70)

# Passenger Class
pclass = int(input("\nPassenger Class (1/2/3): "))

# Gender
print("\nGender:")
print("1. Male")
print("2. Female")

gender = int(input("Enter Choice (1/2): "))

if gender == 1:
    sex = 1
else:
    sex = 0

# Age
age = float(input("\nAge: "))

# Family Details
sibsp = int(input("Number of Siblings/Spouses: "))
parch = int(input("Number of Parents/Children: "))

# Fare
fare = float(input("Fare: "))

# Embarked
print("\nEmbarked Port:")
print("1. Southampton (S)")
print("2. Cherbourg (C)")
print("3. Queenstown (Q)")

embarked_choice = int(input("Enter Choice (1/2/3): "))

if embarked_choice == 1:
    embarked = 2
elif embarked_choice == 2:
    embarked = 0
else:
    embarked = 1

sample = [[
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked
]]

prediction = model.predict(sample)

print("\n" + "=" * 60)

if prediction[0] == 1:
    print("🎉 Prediction : Passenger SURVIVED")
else:
    print("❌ Prediction : Passenger DID NOT SURVIVE")

print("=" * 60)

print("\n🎉 CODSOFT TASK 1 COMPLETED SUCCESSFULLY!")