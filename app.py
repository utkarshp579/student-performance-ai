import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction System")

# ---------------- LOAD MODEL ----------------
with open("student_model.pkl", "rb") as file:
    model, accuracy = pickle.load(file)

st.info(f"📊 Model Accuracy (R² Score): {accuracy:.2f}")

# ======================================================
# 🔹 MANUAL STUDENT PREDICTION (NO GRAPHS, NO GRAPH TITLE)
# ======================================================
st.divider()
st.header("🧑‍🎓 Manual Student Prediction")

study_hours = st.slider("📚 Study Hours per Day", 0, 10, 5)
attendance = st.slider("🏫 Attendance (%)", 40, 100, 75)
previous_marks = st.slider("📝 Previous Exam Marks", 0, 100, 60)
assignments = st.slider("📂 Assignments Score", 0, 100, 65)

if st.button("🚀 Predict Performance"):
    manual_df = pd.DataFrame(
        [[study_hours, attendance, previous_marks, assignments]],
        columns=["StudyHours", "Attendance", "PreviousMarks", "Assignments"]
    )

    prediction = model.predict(manual_df)[0]

    st.success(f"🎯 Predicted Final Marks: {prediction:.2f}")

    if prediction >= 85:
        st.markdown("### 🏆 Performance: Excellent")
    elif prediction >= 70:
        st.markdown("### 👍 Performance: Good")
    elif prediction >= 50:
        st.markdown("### 🙂 Performance: Average")
    else:
        st.markdown("### ⚠️ Performance: Needs Improvement")

# ======================================================
# 🔹 CSV STUDENT PREDICTION (ONLY PLACE WHERE GRAPHS EXIST)
# ======================================================
st.divider()
st.header("📂 CSV Student Prediction (Batch)")

st.caption("Required columns: StudyHours, Attendance, PreviousMarks, Assignments")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"],
    key="csv_upload_single"   # IMPORTANT: unique key prevents duplication
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ["StudyHours", "Attendance", "PreviousMarks", "Assignments"]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV missing required columns.")
        st.stop()

    st.success("✅ CSV uploaded successfully!")
    st.dataframe(df)

    # -------- Predict ALL students --------
    df["PredictedFinalMarks"] = model.predict(df[required_cols]).round(2)

    st.subheader("🤖 Predicted Results (All Students)")
    st.dataframe(df)

    # -------- CSV GRAPHS (ONLY HERE) --------
    st.subheader("📊 Performance Analysis (CSV Data)")

    # Study Hours vs Marks
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["StudyHours"], df["PredictedFinalMarks"])
    ax1.set_xlabel("Study Hours")
    ax1.set_ylabel("Predicted Marks")
    ax1.set_title("Study Hours vs Predicted Marks")
    st.pyplot(fig1)
    plt.close()

  