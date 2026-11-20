"""
Etape 2 bis - Prophet avec jours feries et regresseurs exogenes
Projet 37 - BI2-G9
Memes dates de coupure que forecast_statsforecast.py (recalculees ici : pas besoin de l'avoir lance).
Diapositives alimentees : 15 (syntaxe Prophet), 18 (resultats), 12 (vitesse, via timing.csv)

A lancer depuis la racine du projet :
    python src\\forecast_prophet.py --limit 3     (test rapide sur 3 series)
    python src\\forecast_prophet.py               (les 15 series, modele "Prophet")
    python src\\forecast_prophet.py --no-exog     (variante sans regresseurs, modele "Prophet_noexog")

Entrees : data/sales_weekly.csv, data/holidays.csv, data/raw/features.csv
Sorties : outputs/cv_prophet.csv (ou cv_prophet_noexog.csv)
          outputs/forecast_prophet.csv (ou forecast_prophet_noexog.csv)
          outputs/timing.csv
"""
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

DATA = Path("data/sales_weekly.csv")
HOLIDAYS = Path("data/holidays.csv")
FEATURES = Path("data/raw/features.csv")
OUT = Path("outputs")
FREQ = "W-FRI"
H = 4
REGRESSORS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]

# Prophet (via cmdstanpy) affiche beaucoup de messages : on les coupe.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


def load_sales(limit=None):
    df = pd.read_csv(DATA, parse_dates=["ds"], dtype={"unique_id": str})
    if limit:
        keep = sorted(df["unique_id"].unique())[:limit]
        df = df[df["unique_id"].isin(keep)]
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def add_regressors(sales):
    """Ajoute Temperature, Fuel_Price, CPI, Unemployment du magasin (unique_id = Store_Dept)."""
    feats = pd.read_csv(FEATURES, parse_dates=["Date"]).rename(columns={"Date": "ds"})
    feats = feats[["Store", "ds"] + REGRESSORS]
    sales = sales.copy()
    sales["Store"] = sales["unique_id"].str.split("_").str[0].astype(int)
    sales = sales.merge(feats, on=["Store", "ds"], how="left")
    sales[REGRESSORS] = sales.groupby("unique_id")[REGRESSORS].ffill()
    if sales[REGRESSORS].isna().any().any():
        raise SystemExit("Valeurs manquantes dans les regresseurs : verifiez data/raw/features.csv")
    return sales


def cutoffs_for(dates, cv_windows):
    """Memes coupures que StatsForecast : les cv_windows*H dernieres semaines, par pas de H."""
    dates = sorted(pd.to_datetime(pd.Series(dates)).unique())
    n = len(dates)
    return [pd.Timestamp(dates[n - cv_windows * H - 1 + k * H]) for k in range(cv_windows)]


def fit_predict(train, holidays, use_exog, seed=42):
    m = Prophet(holidays=holidays,
                yearly_seasonality=True,
                weekly_seasonality=False,   # donnees hebdomadaires : pas de cycle a l'interieur de la semaine
                daily_seasonality=False,
                uncertainty_samples=1000)   # 1000 scenarios simules pour l'incertitude
    if use_exog:
        for r in REGRESSORS:
            m.add_regressor(r)
    m.fit(train)

    future = pd.DataFrame({"ds": pd.date_range(train["ds"].max(), periods=H + 1, freq=FREQ)[1:]})
    if use_exog:
        last = train.iloc[-1]
        for r in REGRESSORS:
            # On ne connait pas la temperature ou le CPI des semaines futures :
            # on garde la derniere valeur connue (sinon ce serait une fuite de donnees).
            future[r] = last[r]

    point = m.predict(future)["yhat"].to_numpy()          # prevision centrale
    np.random.seed(seed)
    samples = m.predictive_samples(future)["yhat"]          # tableau (H semaines x 1000 scenarios)
    q = np.percentile(samples, [5, 10, 90, 95], axis=1)
    return pd.DataFrame({"ds": future["ds"].to_numpy(), "yhat": point,
                         "lo90": q[0], "lo80": q[1], "hi80": q[2], "hi90": q[3]})


def log_timing(library, step, n_series, seconds):
    path = OUT / "timing.csv"
    row = pd.DataFrame([{"library": library, "step": step, "n_series": n_series,
                         "seconds": round(seconds, 1),
                         "run_at": datetime.now().isoformat(timespec="seconds")}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="nombre de series (test rapide)")
    p.add_argument("--cv-windows", type=int, default=5, help="fenetres du walk-forward")
    p.add_argument("--no-exog", action="store_true", help="sans regresseurs exogenes")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    use_exog = not args.no_exog
    name = "Prophet" if use_exog else "Prophet_noexog"
    holidays = pd.read_csv(HOLIDAYS, parse_dates=["ds"])
    sales = load_sales(args.limit)
    if use_exog:
        sales = add_regressors(sales)
    cols = ["ds", "y"] + (REGRESSORS if use_exog else [])
    cuts = cutoffs_for(sales["ds"], args.cv_windows)
    ids = sorted(sales["unique_id"].unique())
    print(f"{name} | {len(ids)} series | coupures : {[str(c.date()) for c in cuts]}")

    # 1) Backtest walk-forward sur les memes coupures que StatsForecast
    cv_rows = []
    t0 = time.perf_counter()
    for i, uid in enumerate(ids, 1):
        g = sales[sales["unique_id"] == uid].sort_values("ds")
        for c in cuts:
            train = g[g["ds"] <= c][cols]
            test = g[g["ds"] > c][["ds", "y"]].head(H)
            pred = fit_predict(train, holidays, use_exog).merge(test, on="ds", how="left")
            pred.insert(0, "unique_id", uid)
            pred["cutoff"] = c
            cv_rows.append(pred)
        print(f"  [{i}/{len(ids)}] {uid} : backtest OK")
    t_cv = time.perf_counter() - t0
    cv = pd.concat(cv_rows, ignore_index=True)
    cv["model"] = name
    cv = cv[["unique_id", "ds", "cutoff", "y", "model", "yhat", "lo80", "hi80", "lo90", "hi90"]]
    cv.to_csv(OUT / f"cv_{name.lower()}.csv", index=False)
    print(f"Backtest termine en {t_cv:.1f} s -> outputs/cv_{name.lower()}.csv")
    log_timing("prophet", f"walk-forward ({args.cv_windows} fenetres)", len(ids), t_cv)

    # 2) Prevision finale sur tout l'historique
    fc_rows = []
    t0 = time.perf_counter()
    for uid in ids:
        g = sales[sales["unique_id"] == uid].sort_values("ds")
        pred = fit_predict(g[cols], holidays, use_exog)
        pred.insert(0, "unique_id", uid)
        fc_rows.append(pred)
    t_fc = time.perf_counter() - t0
    fc = pd.concat(fc_rows, ignore_index=True)
    fc["model"] = name
    fc = fc[["unique_id", "ds", "model", "yhat", "lo80", "hi80", "lo90", "hi90"]]
    fc.to_csv(OUT / f"forecast_{name.lower()}.csv", index=False)
    print(f"Prevision terminee en {t_fc:.1f} s -> outputs/forecast_{name.lower()}.csv")
    log_timing("prophet", "forecast final", len(ids), t_fc)


if __name__ == "__main__":
    main()
