import pandas as pd
import joblib

# ----------------------------------
# Load trained model
# ----------------------------------
model_path = r"C:\DAV_STUDENT_PREDICTION_MODEL\models\student_model.pkl"
model = joblib.load(model_path)
print("Model loaded successfully!")

# ----------------------------------
# EXAMPLE INPUT (You can change this)
# ----------------------------------
student_data = {
    "Student ID": [1],
    "Age": [20],
    "Gender": ["Female"],
    "Family support for studies .  Scale of (1-5)": [4],
    "Study Time (Hours/Day)": [3],
    "Class Participation": ["Active"],
    "Extracurricular Activity": ["Yes"],
    "Motivation to study (Scale : 1 -10)": [8],
    "Attendance Percentage ( Range of all subjects)": [85],
    "Previous Semester GPA ": [7.5],
    "Assignments submission ": ["On time"],
    "Internal Theory Exam , Subject 1 mark": [38],
    "Internal Theory Exam , Subject 2 mark": [40],
    "Internal Theory Exam , Subject 3 mark": [42],
    "Internal Theory Exam , Subject 4 mark": [39],
    "Screen Time per day(for non study purpose)": [3]
}

# Convert to dataframe
input_df = pd.DataFrame(student_data)

# ----------------------------------
# Make prediction
# ----------------------------------
prediction = model.predict(input_df)[0]
print("\nPredicted Performance Category:", prediction)
