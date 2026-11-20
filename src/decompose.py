import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("data/sales_weekly.csv", parse_dates=["ds"])

rows = []
for uid, g in df.groupby("unique_id"):
    s = g.set_index("ds")["y"].asfreq("W-FRI")
    stat, pval = adfuller(s.dropna())[:2]
    rows.append({"unique_id": uid, "adf_stat": round(stat, 3),
                 "p_value": round(pval, 4),
                 "stationnaire_5pct": pval < 0.05})
res = pd.DataFrame(rows)
res.to_csv("outputs/adf_results.csv", index=False)
print(res.to_string(index=False))

# Graphiques pour la premiere serie
uid = df["unique_id"].iloc[0]
s = df[df["unique_id"] == uid].set_index("ds")["y"].asfreq("W-FRI")

stl = STL(s, period=52, robust=True).fit()
fig = stl.plot()
fig.set_size_inches(10, 8)
fig.suptitle("Decomposition STL - serie " + uid)
fig.savefig("figures/stl_" + uid + ".png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(s, lags=60, ax=ax[0])
plot_pacf(s, lags=60, ax=ax[1], method="ywm")
fig.savefig("figures/acf_pacf_" + uid + ".png", dpi=150, bbox_inches="tight")

print("Figures creees pour la serie", uid)
