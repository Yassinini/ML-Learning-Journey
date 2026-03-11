#Demo for showing plots progress
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


def sort(df, column):
    return df.sort_values(column)
def Filter(x):
    return x.dropna()

def scatter(d,x,y, outliers=True, percentile=95):
    data=Filter(d)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data = data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(12,8))
    plt.scatter(data[x], data[y], alpha=0.2)
    plt.title(f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def bar(d,x,y):
    data=Filter(d)
    plt.figure(figsize=(15,10))
    plt.bar(data[x], data[y])
    plt.title(f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

def line(d, x, y, outliers=True, percentile=95):
    data=Filter(d)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data=data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(15,10))
    plt.plot(data[x], data[y])
    plt.title(f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

d=pd.read_csv(r"Machine_Learning\Plots+Data\attachment_25167_House_Rent_Dataset.csv.csv")
dc=d[(d["Rent"] < 30000) & (d["Size"] < 7000)] 
sort(dc, "Posted On")
#scatter(dc, "Size", "Rent")
#line(dc , "Posted On", "Rent")

path= kagglehub.dataset_download("dmahajanbe23/bmw-global-automotive-sales")
d=pd.read_csv(path + "/bmw_global_sales_2018_2025.csv")
#scatter(d,"BEV_Share", "Units_Sold", True)

path = kagglehub.dataset_download("zkskhurram/breast-cancer-stat-and-aware-dataset-2022-2025")
d=pd.read_csv(path + "/breast_cancer_risk_factors.csv")

#line(d, "Risk_Factor", "Relative_Risk")

p=pd.read_csv(path + "/breast_cancer_survival_by_stage.csv")
line(p, "Stage", "One_Year_Survival_Pct")