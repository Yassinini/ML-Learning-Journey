# Experimental file - exploring cancer datasets with linear regression
# Model performs poorly on risk factors (r2 = -n) due to categorical data and small dataset
# Survival by stage model performs better (r2 = 0.76) - stage is a strong predictor
# Needs revisiting after learning proper ML theory and evaluation techniques
# Built with AI assistance while learning - not yet fully understood independently


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../Plots+Data"))
from Plots import line , scatter
le = LabelEncoder()

path = kagglehub.dataset_download("zkskhurram/breast-cancer-stat-and-aware-dataset-2022-2025")
Breast_cancer=pd.read_csv(path + "/breast_cancer_risk_factors.csv")



x = pd.get_dummies(Breast_cancer["Risk_Factor"])   
y=Breast_cancer[["Relative_Risk"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
p=LinearRegression()
p.fit(x_train, y_train)
print(r2_score(y_test, p.predict(x_test)))

prediction=p.predict(x)
plt.figure(figsize=(12,8))
plt.scatter(Breast_cancer["Risk_Factor"], Breast_cancer["Relative_Risk"], alpha=0.2, label="Actual")
plt.scatter(Breast_cancer["Risk_Factor"], prediction, alpha=0.2, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Relative Risk")
plt.xlabel("Risk Factor")
plt.ylabel("Relative Risk")
plt.xticks(rotation=45)
#plt.show()

print(r2_score(y,prediction))  #-0.002945508100147709
#Model failed because 

Bsurv=pd.read_csv(path + "/breast_cancer_survival_by_stage.csv")

Bsurv["Stage"] = le.fit_transform(Bsurv["Stage"])
x = Bsurv[["Stage"]]
y=Bsurv[["One_Year_Survival_Pct" , "Five_Year_Survival_Pct", "Ten_Year_Survival_Pct"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
m = LinearRegression()
m.fit(x, y)
prediction = m.predict(x)
print(r2_score(y, prediction))

plt.figure(figsize=(12,8))
plt.plot(Bsurv["Stage"], Bsurv["One_Year_Survival_Pct"], label="1 Year Actual")
plt.plot(Bsurv["Stage"], prediction[:, 0], label="1 Year Predicted")
plt.plot(Bsurv["Stage"], Bsurv["Five_Year_Survival_Pct"], label="5 Year Actual")
plt.plot(Bsurv["Stage"], prediction[:, 1], label="5 Year Predicted")
plt.plot(Bsurv["Stage"], Bsurv["Ten_Year_Survival_Pct"], label="10 Year Actual")
plt.plot(Bsurv["Stage"], prediction[:, 2], label="10 Year Predicted")
plt.legend()
plt.title("Stage vs Survival Rate")
plt.xlabel("Stage")
plt.ylabel("Survival %")
plt.show()


