from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

d=pd.read_csv(r'D:\@Code\yayay\Machine_Learning\kagglecom\train.csv')
d['internet_access'] = (d['internet_access'] == 'Yes').astype(int)
d["facility_rating"] = d["facility_rating"].map({"high": 1, "medium": 0, "low": -1})
d["exam_difficulty"] = d["exam_difficulty"].map({"easy" : 1, "moderate" : 0, "hard" : -1})
x=d[['age', 'study_hours', 'class_attendance', 'sleep_hours', 'internet_access', 'facility_rating', 'exam_difficulty']]
y=d['exam_score'] 

m = RandomForestRegressor(n_estimators=100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
m.fit(x_train, y_train)

prediction=m.predict(x_test)
print("overall r2: \n", r2_score(y_test, prediction))
t=m.predict([[25, 5, 80, 7, 1, 0, 1]])
print("test prediction: \n", t)
     

plt.figure(figsize=(12,8))
plt.scatter(y_test, prediction, alpha=0.2, color="blue", label="Predicted")
plt.plot([20, 100], [20, 100], 'r--', label="Perfect prediction")
plt.legend()
plt.title("Actual vs Predicted Exam Scores")
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.xticks(rotation=45)
plt.show()
