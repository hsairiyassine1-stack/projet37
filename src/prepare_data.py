import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data")
Path("outputs").mkdir(exist_ok=True)

df = pd.read_csv(RAW / "train.csv", parse_dates=["Date"])
df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
df = df.rename(columns={"Date": "ds", "Weekly_Sales": "y"})

counts = df.groupby("unique_id")["ds"].count()
n_max = counts.max()
complete = counts[counts == n_max].index
df = df[df["unique_id"].isin(complete)]

ids = pd.Series(sorted(df["unique_id"].unique())).sample(15, random_state=42)
df = df[df["unique_id"].isin(ids)]

# AJOUT : on garde une trace des valeurs negatives (retours clients) AVANT de les ramener a 0.
# Ce fichier sert de preuve du nettoyage pour la diapositive 17.
neg = df[df["y"] < 0][["unique_id", "ds", "y"]].assign(action="ramene_a_0")
neg.to_csv("outputs/anomalies.csv", index=False)

df["y"] = df["y"].clip(lower=0)

df = df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
df.to_csv(OUT / "sales_weekly.csv", index=False)

print("Lignes :", len(df), "| Series :", df["unique_id"].nunique())
print("Premiere date :", df["ds"].min().date(), "| Derniere :", df["ds"].max().date())
print("Jour de semaine (4 = vendredi) :", df["ds"].dt.dayofweek.unique())
print("Semaines par serie :", n_max)
print("Valeurs negatives ramenees a 0 :", len(neg), "-> outputs/anomalies.csv")  # AJOUT
print("Series retenues :", sorted(df["unique_id"].unique()))                      # AJOUT
