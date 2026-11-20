# Walmart Recruiting - Store Sales Forecasting
# Placez ce fichier dans le MEME dossier que train.csv, test.csv, features.csv, stores.csv
# Dans VS Code : cliquez sur "Run Cell" au-dessus de chaque "# %%" (extension Python + Jupyter),
# ou lancez tout le fichier avec : python walmart_forecast.py
# Installation si besoin : pip install pandas numpy scikit-learn matplotlib

# %% 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# %% 2. Chargement des données
train = pd.read_csv("train.csv", parse_dates=["Date"])
test = pd.read_csv("test.csv", parse_dates=["Date"])
features = pd.read_csv("features.csv", parse_dates=["Date"])
stores = pd.read_csv("stores.csv")

print(train.shape, test.shape, features.shape, stores.shape)

# %% 3. Nettoyage de features.csv puis fusion
md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
features = features.sort_values(["Store", "Date"])
features[md_cols] = features[md_cols].fillna(0)
# CPI et Unemployment manquent sur les dernières semaines : on propage la dernière valeur connue
features[["CPI", "Unemployment"]] = features.groupby("Store")[["CPI", "Unemployment"]].ffill()

features = features.drop(columns="IsHoliday")  # déjà présent dans train/test

df_train = train.merge(features, on=["Store", "Date"], how="left").merge(stores, on="Store", how="left")
df_test = test.merge(features, on=["Store", "Date"], how="left").merge(stores, on="Store", how="left")

# %% 4. Exploration rapide
print(df_train.info())
print(df_train.describe())
print("Ventes négatives :", (df_train["Weekly_Sales"] < 0).sum())

weekly_total = df_train.groupby("Date")["Weekly_Sales"].sum()
weekly_total.plot(figsize=(12, 4), title="Ventes hebdomadaires totales")
plt.show()

df_train.groupby("Type")["Weekly_Sales"].mean().plot(kind="bar", title="Ventes moyennes par type de magasin")
plt.show()

# %% 5. Création des variables
for d in (df_train, df_test):
    d["Year"] = d["Date"].dt.year
    d["Month"] = d["Date"].dt.month
    d["Week"] = d["Date"].dt.isocalendar().week.astype(int)
    d["IsHoliday"] = d["IsHoliday"].astype(int)
    d["Type"] = d["Type"].map({"A": 0, "B": 1, "C": 2})

cols = ["Store", "Dept", "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "Type", "Size", "Year", "Month", "Week"] + md_cols

# %% 6. Séparation entraînement / validation (par date, pas au hasard)
cutoff = "2012-06-01"
tr = df_train[df_train["Date"] < cutoff]
va = df_train[df_train["Date"] >= cutoff]
print("Train :", tr.shape, "| Validation :", va.shape)

# %% 7. Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(tr[cols], tr["Weekly_Sales"])

# %% 8. Évaluation avec la métrique WMAE (semaines de fête pondérées x5)
def wmae(y_true, y_pred, is_holiday):
    w = np.where(is_holiday == 1, 5, 1)
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)

pred_va = model.predict(va[cols])
print("WMAE validation :", wmae(va["Weekly_Sales"].values, pred_va, va["IsHoliday"].values))

importances = pd.Series(model.feature_importances_, index=cols).sort_values()
importances.plot(kind="barh", figsize=(8, 6), title="Importance des variables")
plt.show()

# %% 9. Réentraînement sur toutes les données puis prédiction sur test.csv
model.fit(df_train[cols], df_train["Weekly_Sales"])
df_test["Weekly_Sales"] = model.predict(df_test[cols])

# %% 10. Création du fichier de soumission
df_test["Id"] = (df_test["Store"].astype(str) + "_" + df_test["Dept"].astype(str)
                 + "_" + df_test["Date"].dt.strftime("%Y-%m-%d"))
df_test[["Id", "Weekly_Sales"]].to_csv("submission.csv", index=False)
print("submission.csv créé :", df_test.shape[0], "lignes")
