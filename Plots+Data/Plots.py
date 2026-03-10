#Demo for showing plots progress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d=pd.read_csv(r"Plots+Data\attachment_25167_House_Rent_Dataset.csv.csv")
dc=d[(d["Rent"] < 30000) & (d["Size"] < 7000)]
sorted_dates=d.sort_values("Posted On")

def Filter_NaN(x):
    return x.dropna()

def Scatter_Housing():
    data= Filter_NaN(dc)
    plt.figure(figsize=(15,10))
    plt.scatter(data["Size"], data["Rent"], alpha=0.2)
    plt.title("Size to Rent")
    plt.xlabel("Size")
    plt.ylabel("Rent")
    plt.xticks(rotation=45)
    plt.show()

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

Line_Housing()
