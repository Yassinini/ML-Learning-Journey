from Plots import scatter, bar, line, Filter, sort
import kagglehub
import pandas as pd


path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
d = pd.read_csv(path + "/Housing.csv")
scatter(d, "area", "price")
#line(d, "area", "price")


path= kagglehub.dataset_download("dmahajanbe23/bmw-global-automotive-sales")
d=pd.read_csv(path + "/bmw_global_sales_2018_2025.csv")
#scatter(d,"BEV_Share", "Units_Sold", True)

path = kagglehub.dataset_download("zkskhurram/breast-cancer-stat-and-aware-dataset-2022-2025")
d=pd.read_csv(path + "/breast_cancer_risk_factors.csv")
#line(d, "Risk_Factor", "Relative_Risk")

p=pd.read_csv(path + "/breast_cancer_survival_by_stage.csv")
#line(p, "Stage", "One_Year_Survival_Pct")
#line(p, "Stage", "Five_Year_Survival_Pct")
#line(p, "Stage", "Ten_Year_Survival_Pct")
#line(p, "Stage", "Typical_Treatment")

path = kagglehub.dataset_download("alexisbcook/data-for-datavis")
l=pd.read_csv(path + "/insurance.csv")
#scatter(l, "age", "bmi", title="Age vs BMI, Insurance")