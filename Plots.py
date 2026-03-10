#Demo for showing plots progress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d=pd.read_csv(r"D:\@Code\yayay\PY\Machine Learning\attachment_25167_House_Rent_Dataset.csv.csv")
dc=d[(d["Rent"] < 30000) & (d["Size"] < 7000)]
#print(d.isna().sum()) none

def Filter_NaN(x):
    return x.dropna()

def Scatter_Housing():
    dc= Filter_NaN(dc)
    plt.figure(figsize=(15,10))
    plt.scatter(dc["Size"], dc["Rent"], alpha=0.2)
    plt.title("Size to Rent")
    plt.xlabel("Size")
    plt.ylabel("Rent")
    plt.show()

def Bar_Housing():
    data=Filter_NaN(d)
    plt.figure(figsize=(15,10))
    plt.bar(data["City"], data["Rent"])
    plt.title("City to Rent")
    plt.xlabel("City")
    plt.ylabel("Rent")
    plt.show()
