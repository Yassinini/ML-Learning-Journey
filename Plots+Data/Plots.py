#Demo for showing plots progress
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


def sort(df, column):
    return df.sort_values(column)
def Filter(x):
    return x.dropna()

def scatter(d,x,y, outliers=True, percentile=95, title=""):
    data=Filter(d)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data = data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(12,8))
    plt.scatter(data[x], data[y], alpha=0.2)
    plt.title(title if title 
              else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def bar(d,x,y, title=""):
    data=Filter(d)
    plt.figure(figsize=(15,10))
    plt.bar(data[x], data[y])
    plt.title(title if title 
              else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def line(d, x, y, outliers=True, percentile=95 , title=""):
    data=Filter(d)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data=data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(15,10))
    plt.plot(data[x], data[y])
    plt.title(title if title 
              else f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()


path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
d = pd.read_csv(path + "/Housing.csv")
#scatter(d, "area", "price")
#line(d, "area", "price")


path= kagglehub.dataset_download("dmahajanbe23/bmw-global-automotive-sales")
d=pd.read_csv(path + "/bmw_global_sales_2018_2025.csv")
#scatter(d,"BEV_Share", "Units_Sold", True)

path = kagglehub.dataset_download("zkskhurram/breast-cancer-stat-and-aware-dataset-2022-2025")
d=pd.read_csv(path + "/breast_cancer_risk_factors.csv")

#line(d, "Risk_Factor", "Relative_Risk")

p=pd.read_csv(path + "/breast_cancer_survival_by_stage.csv")
line(p, "Stage", "One_Year_Survival_Pct")
line(p, "Stage", "Five_Year_Survival_Pct")
line(p, "Stage", "Ten_Year_Survival_Pct")
#line(p, "Stage", "Typical_Treatment")

path = kagglehub.dataset_download("alexisbcook/data-for-datavis")
l=pd.read_csv(path + "/insurance.csv")
#scatter(l, "age", "bmi", title="Age vs BMI, Insurance")