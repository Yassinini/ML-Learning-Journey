from Plots import scatter, bar, line, Filter, sort
import kagglehub
import pandas as pd


path = kagglehub.dataset_download("anassarfraz13/housing-dataset-info-about-houses")
housing = pd.read_csv(path + "/Housing.csv")
#scatter(housing, "area", "price")
#line(housing, "area", "price")


path= kagglehub.dataset_download("dmahajanbe23/bmw-global-automotive-sales")
bmw=pd.read_csv(path + "/bmw_global_sales_2018_2025.csv")
#scatter(bmw,"BEV_Share", "Units_Sold", True)

path = kagglehub.dataset_download("zkskhurram/breast-cancer-stat-and-aware-dataset-2022-2025")
Breast_cancer=pd.read_csv(path + "/breast_cancer_risk_factors.csv")
#line(Breast_cancer, "Risk_Factor", "Relative_Risk")

Bcancer_Survival=pd.read_csv(path + "/breast_cancer_survival_by_stage.csv")
#line(Bcancer_Survival, "Stage", "One_Year_Survival_Pct")
#line(Bcancer_Survival, "Stage", "Five_Year_Survival_Pct")
#line(Bcancer_Survival, "Stage", "Ten_Year_Survival_Pct")
#line(Bcancer_Survival, "Stage", "Typical_Treatment")

path = kagglehub.dataset_download("alexisbcook/data-for-datavis")
Insurance=pd.read_csv(path + "/insurance.csv")
#scatter(Insurance, "age", "bmi", title="Age vs BMI, Insurance")