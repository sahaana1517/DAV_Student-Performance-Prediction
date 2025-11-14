import pandas as pd

# ---------------------------------
# Step 1: Load raw dataset
# ---------------------------------
df = pd.read_csv(r"C:\DAV_STUDENT_PREDICTION_MODEL\data\RAW DATA DAV_final.csv")
print("Raw data loaded successfully!")

# ---------------------------------
# Step 2: Calculate internal exam average
# ---------------------------------
df["internal_avg"] = (
    df["Internal Theory Exam , Subject 1 mark"] +
    df["Internal Theory Exam , Subject 2 mark"] +
    df["Internal Theory Exam , Subject 3 mark"] +
    df["Internal Theory Exam , Subject 4 mark"]
) / 4

# ---------------------------------
# Step 3: Calculate performance_score
# (Weighted score used for category)
# ---------------------------------
df["performance_score"] = (
    (df["Previous Semester GPA "] * 0.4) +
    (df["internal_avg"] * 0.4) +
    (df["Attendance Percentage ( Range of all subjects)"] * 0.2)
)

# ---------------------------------
# Step 4: Convert performance score to category
# ---------------------------------
high_threshold = df["performance_score"].quantile(0.66)
medium_threshold = df["performance_score"].quantile(0.33)

def convert(score):
    if score >= high_threshold:
        return "High"
    elif score >= medium_threshold:
        return "Medium"
    else:
        return "Low"

df["Performance_Category"] = df["performance_score"].apply(convert)

# ---------------------------------
# Step 5: Save new dataset with target
# ---------------------------------
output_path = r"C:\DAV_STUDENT_PREDICTION_MODEL\data\dataset_with_target.csv"
df.to_csv(output_path, index=False)

print("Target column created successfully!")
print("Saved to:", output_path)
print(df[["performance_score", "Performance_Category"]].head())
