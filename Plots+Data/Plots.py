#Demo for showing plots progress
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

d=pd.read_csv(r"Plots+Data\attachment_25167_House_Rent_Dataset.csv.csv")
dc=d[(d["Rent"] < 30000) & (d["Size"] < 7000)]
sorted_dates=d.sort_values("Posted On")

def Filter(x):
    return x.dropna()
def scatter(d,x,y, outliers=True, percentile=95):
    data=Filter(d)
    if outliers and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
        data = data[(data[x] < np.percentile(data[x], percentile)) & 
                    (data[y] < np.percentile(data[y], percentile))]
    plt.figure(figsize=(15,10))
    plt.scatter(data[x], data[y], alpha=0.2)
    plt.title(f"{x} to {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.show()

#scatter(dc, "Size", "Rent", True)

path= kagglehub.dataset_download("dmahajanbe23/bmw-global-automotive-sales")
d=pd.read_csv(path + "/bmw_global_sales_2018_2025.csv")
scatter(d,"BEV_Share", "Units_Sold", True)

def Bar_Housing():
    data=Filter_NaN(d)
    plt.figure(figsize=(15,10))
    plt.bar(data["City"], data["Rent"])
    plt.title("City to Rent")
    plt.xlabel("City")
    plt.ylabel("Rent")
    plt.xticks(rotation=45)
    plt.show()

def Line_Housing():
    data=Filter_NaN(d)
    plt.figure(figsize=(15,10))
    plt.plot(sorted_dates["Posted On"][0:20], sorted_dates["Rent"][0:20])
    plt.title(" Date Posted to Rent")
    plt.xlabel("Date Posted")
    plt.ylabel("Rent")
    plt.xticks(rotation=45)
    plt.show()