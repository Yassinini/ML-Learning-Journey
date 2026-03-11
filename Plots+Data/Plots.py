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

if __name__ == "__main__":
    pass