# ============================================================
# STUDENT PERFORMANCE AI - STREAMLIT WEB APP (LINE BY LINE EXPLANATION)
# ============================================================

# ----- IMPORTS -----

import streamlit as st
# Imports Streamlit library for creating interactive web applications.
# Streamlit allows building data apps with simple Python code - no HTML/CSS/JS needed.

import pickle
# Imports pickle module for deserializing (loading) saved Python objects.
# Used to load the pre-trained model from a file.

import pandas as pd
# Imports pandas for data manipulation and handling DataFrames.
# Used for processing CSV files and structuring prediction inputs.

import matplotlib.pyplot as plt
# Imports matplotlib for creating static visualizations/graphs.
# Used to plot scatter charts for performance analysis.


# ----- PAGE CONFIGURATION -----

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="centered"
)
# Configures the Streamlit page settings:
# - page_title: Text shown in browser tab
# - page_icon: Emoji or image shown in browser tab
# - layout: "centered" keeps content in the middle (vs "wide" for full width)

st.title("🎓 Student Performance Prediction System")
# Displays the main title/heading of the web app.
# st.title() creates the largest text size in Streamlit.


# ----- LOAD TRAINED MODEL -----

with open("student_model.pkl", "rb") as file:
    model, accuracy = pickle.load(file)
# Opens the saved model file in read-binary mode ("rb").
# pickle.load() deserializes and loads the saved tuple (model, accuracy).
# 'model' is the trained LinearRegression object.
# 'accuracy' is the R² score saved during training.

st.info(f"📊 Model Accuracy (R² Score): {accuracy:.2f}")
# Displays an info box (blue) showing the model's accuracy.
# st.info() creates a styled information message.
# {accuracy:.2f} formats the number to 2 decimal places.


# ======================================================
# 🔹 SECTION 1: MANUAL STUDENT PREDICTION
# ======================================================

st.markdown("---")
# Renders a horizontal line (divider) using Markdown syntax.
# st.markdown() allows rendering Markdown-formatted text.

st.header("🧑‍🎓 Manual Student Prediction")
# Displays a section header (smaller than title, larger than subheader).

study_hours = st.slider("📚 Study Hours per Day", 0, 10, 5)
# Creates an interactive slider widget for Study Hours input.
# Parameters: label, min_value=0, max_value=10, default_value=5
# Returns the current slider value selected by user.

attendance = st.slider("🏫 Attendance (%)", 40, 100, 75)
# Slider for Attendance percentage.
# Range: 40% to 100%, default: 75%

previous_marks = st.slider("📝 Previous Exam Marks", 0, 100, 60)
# Slider for Previous Marks input.
# Range: 0 to 100, default: 60

assignments = st.slider("📂 Assignments Score", 0, 100, 65)
# Slider for Assignments Score input.
# Range: 0 to 100, default: 65

if st.button("🚀 Predict Performance"):
    # st.button() creates a clickable button.
    # Returns True when clicked, False otherwise.
    # Code inside this block runs only when button is clicked.

    manual_df = pd.DataFrame(
        [[study_hours, attendance, previous_marks, assignments]],
        columns=["StudyHours", "Attendance", "PreviousMarks", "Assignments"]
    )
    # Creates a DataFrame with single row containing user inputs.
    # [[...]] creates a 2D array (list of lists) for one row of data.
    # columns= specifies column names matching the trained model's expected features.

    prediction = model.predict(manual_df)[0]
    # Uses the loaded model to predict Final Marks.
    # model.predict() returns an array, [0] gets the first (only) prediction.

    st.success(f"🎯 Predicted Final Marks: {prediction:.2f}")
    # Displays a success box (green) with the prediction result.
    # st.success() creates a styled success message.

    if prediction >= 85:
        st.markdown("### 🏆 Performance: Excellent")
    elif prediction >= 70:
        st.markdown("### 👍 Performance: Good")
    elif prediction >= 50:
        st.markdown("### 🙂 Performance: Average")
    else:
        st.markdown("### ⚠️ Performance: Needs Improvement")
    # Conditional logic to categorize performance based on predicted marks.
    # ### in Markdown creates a level-3 heading.
    # Thresholds: 85+ Excellent, 70-84 Good, 50-69 Average, <50 Needs Improvement.


# ======================================================
# 🔹 SECTION 2: CSV BATCH PREDICTION
# ======================================================

st.markdown("---")
# Another horizontal divider to separate sections.

st.header("📂 CSV Student Prediction (Batch)")
# Section header for CSV upload functionality.

st.caption("Required columns: StudyHours, Attendance, PreviousMarks, Assignments")
# Displays small caption text explaining CSV requirements.
# st.caption() renders smaller, muted text.

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"],
    key="csv_upload_single"
)
# Creates a file upload widget.
# - First argument: Label text shown above the uploader
# - type=["csv"]: Only accepts CSV files
# - key: Unique identifier to prevent widget duplication issues
# Returns UploadedFile object if file uploaded, None otherwise.

if uploaded_file is not None:
    # Checks if a file was uploaded.
    # Code inside runs only when a valid file is present.

    df = pd.read_csv(uploaded_file)
    # Reads the uploaded CSV file into a pandas DataFrame.
    # Streamlit's UploadedFile works directly with pd.read_csv().

    required_cols = ["StudyHours", "Attendance", "PreviousMarks", "Assignments"]
    # List of column names required for prediction.
    # These must match the features the model was trained on.

    if not all(col in df.columns for col in required_cols):
        st.error("❌ CSV missing required columns.")
        st.stop()
    # Validates that all required columns exist in the uploaded CSV.
    # all() returns True only if every column is present.
    # st.error() displays a red error message.
    # st.stop() halts script execution (prevents further processing).

    st.success("✅ CSV uploaded successfully!")
    # Displays success message confirming valid upload.

    st.dataframe(df)
    # Displays the uploaded DataFrame as an interactive table.
    # st.dataframe() allows scrolling, sorting, and searching.

    df["PredictedFinalMarks"] = model.predict(df[required_cols]).round(2)
    # Predicts Final Marks for ALL students in the CSV.
    # df[required_cols] extracts only the feature columns.
    # model.predict() returns predictions for each row.
    # .round(2) rounds to 2 decimal places.
    # Creates a new column "PredictedFinalMarks" with results.

    st.subheader("🤖 Predicted Results (All Students)")
    # Displays a subheader (smaller than header).

    st.dataframe(df)
    # Shows the updated DataFrame with predictions included.

    # ----- VISUALIZATION SECTION -----

    st.subheader("📊 Performance Analysis (CSV Data)")
    # Subheader for the graphs section.

    fig1, ax1 = plt.subplots()
    # Creates a matplotlib figure and axes object.
    # fig1: The figure (canvas) containing the plot.
    # ax1: The axes (actual plot area) where data is drawn.

    ax1.scatter(df["StudyHours"], df["PredictedFinalMarks"])
    # Creates a scatter plot on the axes.
    # X-axis: Study Hours from CSV data.
    # Y-axis: Predicted Final Marks (calculated above).

    ax1.set_xlabel("Study Hours")
    # Sets the label for the X-axis.

    ax1.set_ylabel("Predicted Marks")
    # Sets the label for the Y-axis.

    ax1.set_title("Study Hours vs Predicted Marks")
    # Sets the title displayed above the plot.

    st.pyplot(fig1)
    # Renders the matplotlib figure in the Streamlit app.
    # st.pyplot() integrates matplotlib plots into Streamlit.

    plt.close()
    # Closes the figure to free memory.
    # Best practice when creating multiple plots to avoid memory leaks.
