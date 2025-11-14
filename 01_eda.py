# ==========================
# 01_eda.py - Data Exploration Script
# ==========================

import pandas as pd
from pathlib import Path

# Path to dataset
DATA_PATH = r"C:\DAV_STUDENT_PREDICTION_MODEL\data\RAW DATA DAV_final.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic info
print("✅ Data Loaded Successfully!")
print("Shape of dataset:", df.shape)
print("\nColumns available:\n", df.columns.tolist())

# Data types and missing values
print("\n--- Column Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Unique Values per Column ---")
print(df.nunique())

# Show first few rows
print("\n--- First 5 Rows ---")
print(df.head())

# Quick stats for numeric columns
print("\n--- Statistical Summary ---")
print(df.describe())

# If there are categorical columns, show their value counts
print("\n--- Categorical Columns Distribution ---")
cat_cols = [c for c in df.columns if df[c].dtype == 'object']
for col in cat_cols:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts())
