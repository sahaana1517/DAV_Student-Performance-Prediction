import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model_path = r"C:\DAV_STUDENT_PREDICTION_MODEL\models\student_model.pkl"
model = joblib.load(model_path)

st.title("🎓 Student Performance Prediction Web App")
st.write("Enter the student's details below to predict their performance category.")

# -----------------------------
# Input Form
# -----------------------------
student_id = st.number_input("Student ID", min_value=1, step=1)
age = st.number_input("Age", min_value=10, max_value=30, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
family_support = st.slider("Family support (1–5)", 1, 5)
study_time = st.slider("Study Time (Hours/Day)", 1, 10)
class_participation = st.selectbox("Class Participation", ["Active", "Moderate"])
extra_activity = st.selectbox("Extracurricular Activity", ["Yes", "No"])
motivation = st.slider("Motivation (1–10)", 1, 10)
attendance = st.slider("Attendance Percentage", 0, 100)
previous_gpa = st.number_input("Previous Semester GPA", 0.0, 10.0, format="%.2f")
assignment_submit = st.selectbox("Assignments submission", ["On time", "Delayed", "Both"])
sub1 = st.number_input("Internal Exam Subject 1 Marks", 0, 50)
sub2 = st.number_input("Internal Exam Subject 2 Marks", 0, 50)
sub3 = st.number_input("Internal Exam Subject 3 Marks", 0, 50)
sub4 = st.number_input("Internal Exam Subject 4 Marks", 0, 50)
screen_time = st.number_input("Screen Time per day", 0, 15)

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame({
    "Student ID": [student_id],
    "Age": [age],
    "Gender": [gender],
    "Family support for studies .  Scale of (1-5)": [family_support],
    "Study Time (Hours/Day)": [study_time],
    "Class Participation": [class_participation],
    "Extracurricular Activity": [extra_activity],
    "Motivation to study (Scale : 1 -10)": [motivation],
    "Attendance Percentage ( Range of all subjects)": [attendance],
    "Previous Semester GPA ": [previous_gpa],
    "Assignments submission ": [assignment_submit],
    "Internal Theory Exam , Subject 1 mark": [sub1],
    "Internal Theory Exam , Subject 2 mark": [sub2],
    "Internal Theory Exam , Subject 3 mark": [sub3],
    "Internal Theory Exam , Subject 4 mark": [sub4],
    "Screen Time per day(for non study purpose)": [screen_time],
})

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Performance"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Performance Category: **{prediction}**")
