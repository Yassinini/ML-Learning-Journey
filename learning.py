from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import kagglehub

path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
housing = pd.read_csv(path + "/Housing.csv")

x=housing[["area", "bedrooms", "bathrooms", "stories", "parking", "mainroad"]]
y=housing[["price"]]

m = LinearRegression()
m.fit(x,y)

print(m.predict([[1900, 3, 2, 2, 2, 1]]))
print(r2_score(y, m.predict(x)))
