import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv("dataset.csv")

X = data[['StudyHours', 'Attendance', 'PreviousMarks', 'Assignments']]
y = data['FinalMarks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

with open("student_model.pkl", "wb") as file:
    pickle.dump((model, accuracy), file)

print(f"✅ Model trained successfully")
print(f"📊 Model Accuracy (R² Score): {accuracy:.2f}")
