"""
Etape 2 - StatsForecast : validation walk-forward + intervalles conformes
Projet 37 - BI2-G9
Diapositives alimentees : 14 (syntaxe), 18 (resultats), 12 (vitesse, via timing.csv)

A lancer depuis la racine du projet (dossier projet37) :
    python src\\forecast_statsforecast.py --limit 3    (test rapide sur 3 series)
    python src\\forecast_statsforecast.py              (les 15 series)
    python src\\forecast_statsforecast.py --fast       (AutoARIMA avec une recherche reduite, si trop lent)

Entree  : data/sales_weekly.csv  (unique_id, ds, y)
Sorties : outputs/cv_statsforecast.csv        backtest walk-forward, format long
          outputs/forecast_statsforecast.csv  prevision des 4 semaines suivantes, format long
          outputs/timing.csv                  temps de calcul (une ligne ajoutee a chaque etape)
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive
from statsforecast.utils import ConformalIntervals

DATA = Path("data/sales_weekly.csv")
OUT = Path("outputs")
FREQ = "W-FRI"     # les dates Walmart sont des vendredis
H = 4              # horizon : 4 semaines
SEASON = 52        # saisonnalite annuelle en donnees hebdomadaires
LEVELS = [80, 90]  # 80 % -> P10/P90 ; 90 % -> P05/P95


def load(limit=None):
    df = pd.read_csv(DATA, parse_dates=["ds"], dtype={"unique_id": str})
    if limit:
        keep = sorted(df["unique_id"].unique())[:limit]
        df = df[df["unique_id"].isin(keep)]
    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def build_models(calib_windows, fast=False):
    # Intervalles conformes : on mesure les erreurs reelles du modele sur
    # `calib_windows` fenetres de H semaines a la fin de chaque historique d'entrainement.
    ci = ConformalIntervals(h=H, n_windows=calib_windows)
    arima_opts = {"max_P": 1, "max_Q": 1} if fast else {}
    return [
        SeasonalNaive(season_length=SEASON, prediction_intervals=ci),  # reference : "meme semaine l'an dernier"
        AutoARIMA(season_length=SEASON, prediction_intervals=ci, **arima_opts),
        AutoETS(season_length=SEASON, prediction_intervals=ci),
        AutoTheta(season_length=SEASON, prediction_intervals=ci),
    ]


def to_long(wide, with_truth):
    """Format large de StatsForecast (une colonne par modele) -> format long commun a tous les modeles."""
    if "unique_id" not in wide.columns:  # selon la version, unique_id peut etre l'index
        wide = wide.reset_index()
    base = ["unique_id", "ds"] + (["cutoff", "y"] if with_truth else [])
    models = [c for c in wide.columns
              if c not in base and c != "index" and "-lo-" not in c and "-hi-" not in c]
    parts = []
    for m in models:
        part = wide[base].copy()
        part["model"] = m
        part["yhat"] = wide[m]                      # prevision centrale (utilisee comme P50)
        for lv in LEVELS:
            part[f"lo{lv}"] = wide[f"{m}-lo-{lv}"]  # lo80 = P10, lo90 = P05
            part[f"hi{lv}"] = wide[f"{m}-hi-{lv}"]  # hi80 = P90, hi90 = P95
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


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
    p.add_argument("--calib-windows", type=int, default=4, help="fenetres de calibration conforme")
    p.add_argument("--fast", action="store_true", help="AutoARIMA avec une recherche reduite")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    df = load(args.limit)
    n_series = df["unique_id"].nunique()
    n_weeks = df.groupby("unique_id").size().min()
    first_train = n_weeks - args.cv_windows * H
    calib_train = first_train - args.calib_windows * H
    print(f"{n_series} series | {n_weeks} semaines par serie")
    print(f"Walk-forward : {args.cv_windows} fenetres de {H} semaines = {args.cv_windows * H} semaines de test")
    print(f"1re fenetre : entrainement sur {first_train} semaines, "
          f"dont calibration conforme sur les {args.calib_windows * H} dernieres")
    if calib_train < 2 * SEASON:
        print(f"ATTENTION : certains ajustements n'ont que {calib_train} semaines (< 2 ans) : "
              "la saisonnalite 52 sera estimee sur peu de donnees.")

    sf = StatsForecast(models=build_models(args.calib_windows, args.fast), freq=FREQ, n_jobs=-1)

    # 1) Backtest walk-forward : on coupe, on prevoit 4 semaines, on compare, on avance de 4 semaines.
    t0 = time.perf_counter()
    cv = sf.cross_validation(df=df, h=H, step_size=H, n_windows=args.cv_windows, level=LEVELS)
    t_cv = time.perf_counter() - t0
    cv_long = to_long(cv, with_truth=True)
    cv_long.to_csv(OUT / "cv_statsforecast.csv", index=False)
    cutoffs = sorted(pd.to_datetime(cv_long["cutoff"]).dt.date.unique())
    print(f"Backtest termine en {t_cv:.1f} s -> outputs/cv_statsforecast.csv")
    print("Dates de coupure :", [str(c) for c in cutoffs])
    log_timing("statsforecast", f"cross_validation ({args.cv_windows} fenetres)", n_series, t_cv)

    # 2) Prevision "pour de vrai" : on reentraine sur tout l'historique et on prevoit les 4 semaines suivantes.
    t0 = time.perf_counter()
    fc = sf.forecast(df=df, h=H, level=LEVELS)
    t_fc = time.perf_counter() - t0
    fc_long = to_long(fc, with_truth=False)
    fc_long.to_csv(OUT / "forecast_statsforecast.csv", index=False)
    weeks = sorted(pd.to_datetime(fc_long["ds"]).dt.date.unique())
    print(f"Prevision terminee en {t_fc:.1f} s -> outputs/forecast_statsforecast.csv")
    print("Semaines prevues :", [str(w) for w in weeks])
    log_timing("statsforecast", "forecast final", n_series, t_fc)


if __name__ == "__main__":  # obligatoire sous Windows avec n_jobs=-1 (calcul en parallele)
    main()
