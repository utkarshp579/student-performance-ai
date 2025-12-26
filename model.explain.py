'''Section   	Purpose
Imports        	Load libraries for data handling, ML model, evaluation, and saving
Data Loading 	Read student data from CSV into a DataFrame
Feature Selection	Choose 4 input features (StudyHours, Attendance, PreviousMarks, Assignments) and 1 target (FinalMarks)
Data Splitting	80% training, 20% testing with reproducible split
Training	    Fit a Linear Regression model to learn patterns
Evaluation	    Calculate R² score to measure prediction accuracy
Saving	        Serialize model + accuracy to a .pkl file for reuse
Output	        Display success message and accuracy score'''



# ============================================================
# STUDENT PERFORMANCE PREDICTION MODEL - LINE BY LINE EXPLANATION
# ============================================================

# ----- IMPORTS -----

import pandas as pd
# Imports pandas library for data manipulation and analysis.
# Pandas provides DataFrame structures to work with tabular data like CSV files.

from sklearn.model_selection import train_test_split
# Imports the train_test_split function from scikit-learn.
# This function splits data into training and testing sets for model validation.

from sklearn.linear_model import LinearRegression
# Imports the LinearRegression class from scikit-learn.
# Linear Regression is a supervised learning algorithm that predicts continuous values.

from sklearn.metrics import r2_score
# Imports the r2_score function to evaluate model performance.
# R² (coefficient of determination) measures how well the model fits the data (0 to 1, higher is better).

import pickle
# Imports pickle module for serializing Python objects.
# Used to save the trained model to a file for later use.


# ----- DATA LOADING -----

data = pd.read_csv("dataset.csv")
# Reads the CSV file "dataset.csv" into a pandas DataFrame.
# The DataFrame 'data' now contains all rows and columns from the CSV file.


# ----- FEATURE AND TARGET SELECTION -----

X = data[['StudyHours', 'Attendance', 'PreviousMarks', 'Assignments']]
# Selects the feature columns (independent variables) that will be used to make predictions.
# X contains 4 features: StudyHours, Attendance, PreviousMarks, and Assignments.
# These are the inputs the model will learn from.

y = data['FinalMarks']
# Selects the target column (dependent variable) that we want to predict.
# y contains the FinalMarks - this is what the model will try to predict.


# ----- DATA SPLITTING -----

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Splits the data into training (80%) and testing (20%) sets.
# - X_train, y_train: Data used to train the model
# - X_test, y_test: Data used to evaluate model performance
# - test_size=0.2: 20% of data goes to testing, 80% to training
# - random_state=42: Ensures reproducible results (same split every time)


# ----- MODEL TRAINING -----

model = LinearRegression()
# Creates an instance of the LinearRegression model.
# This initializes an untrained linear regression model object.

model.fit(X_train, y_train)
# Trains the model using the training data.
# The model learns the relationship between features (X_train) and target (y_train).
# It finds the best-fit line by minimizing the sum of squared errors.


# ----- MODEL EVALUATION -----

y_pred = model.predict(X_test)
# Uses the trained model to make predictions on the test data.
# y_pred contains the predicted FinalMarks for the test set.

accuracy = r2_score(y_test, y_pred)
# Calculates the R² score by comparing actual values (y_test) with predictions (y_pred).
# R² = 1 means perfect prediction, R² = 0 means model is no better than predicting the mean.
# Negative R² means the model performs worse than a horizontal line.


# ----- MODEL SAVING -----

with open("student_model.pkl", "wb") as file:
    pickle.dump((model, accuracy), file)
# Opens a file called "student_model.pkl" in write-binary mode ("wb").
# pickle.dump() serializes and saves both the model and its accuracy as a tuple.
# This allows loading the trained model later without retraining.
# The 'with' statement ensures the file is properly closed after writing.


# ----- OUTPUT -----

print(f"✅ Model trained successfully")
# Prints a success message indicating the model training is complete.

print(f"📊 Model Accuracy (R² Score): {accuracy:.2f}")
# Prints the model's R² score formatted to 2 decimal places.
# Example output: "📊 Model Accuracy (R² Score): 0.85"
