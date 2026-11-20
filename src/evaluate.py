"""
Etape 2 ter - Evaluation, couverture des intervalles et choix du meilleur modele par serie
Projet 37 - BI2-G9
Diapositives alimentees : 18 (tableau de resultats) ; 10 et 19 via outputs/forecasts.csv

A lancer depuis la racine du projet, APRES les scripts de prevision :
    python src\\evaluate.py

Entrees : outputs/cv_*.csv, outputs/forecast_*.csv, data/sales_weekly.csv, data/holidays.csv
Sorties : outputs/metrics.csv          une ligne par (serie, modele)  -> contrat P2 -> P3
          outputs/metrics_summary.csv  une ligne par modele           -> diapositive 18
          outputs/forecasts.csv        P05..P95 par semaine future     -> contrat P2 -> P3

Definitions (a reprendre a l'oral) :
  sMAPE       = moyenne de 2|y - yhat| / (|y| + |yhat|), en %  (0 % = parfait, 200 % = maximum)
  MASE        = erreur absolue moyenne / erreur moyenne du "naif saisonnier" (meme semaine l'an dernier)
                calculee sur l'historique AVANT la 1re coupure. MASE < 1 : mieux que le naif saisonnier.
  RMSE        = racine de la moyenne des erreurs au carre (en dollars de ventes ; penalise les grosses erreurs)
  coverage_80 = % des semaines de test ou la vraie vente tombe dans [P10 ; P90] (cible : environ 80 %)
  coverage_90 = % des semaines de test ou la vraie vente tombe dans [P05 ; P95] (cible : environ 90 %)
  width_80    = largeur moyenne de [P10 ; P90] : une couverture elevee avec des intervalles enormes n'aide pas
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
DATA = Path("data/sales_weekly.csv")
HOLIDAYS = Path("data/holidays.csv")
SEASON = 52


def read_all(pattern):
    files = sorted(OUT.glob(pattern))
    if not files:
        raise SystemExit(f"Aucun fichier {pattern} dans outputs/ : lancez d'abord les scripts de prevision.")
    print(f"Fichiers lus ({pattern}) :", [f.name for f in files])
    dfs = [pd.read_csv(f, dtype={"unique_id": str}) for f in files]
    return pd.concat(dfs, ignore_index=True)


def mase_scales(sales, first_cutoff):
    """Echelle du MASE : erreur moyenne du naif saisonnier, uniquement sur le passe (pas de fuite)."""
    scales = {}
    for uid, g in sales[sales["ds"] <= first_cutoff].groupby("unique_id"):
        y = g.sort_values("ds")["y"].to_numpy()
        diff = np.abs(y[SEASON:] - y[:-SEASON])
        scales[uid] = diff.mean() if len(diff) else np.nan
    return scales


def series_metrics(g, scale):
    err = g["y"] - g["yhat"]
    denom = g["y"].abs() + g["yhat"].abs()
    smape_terms = np.where(denom == 0, 0.0, 2 * err.abs() / denom.replace(0, np.nan))
    in80 = (g["y"] >= g["lo80"]) & (g["y"] <= g["hi80"])
    in90 = (g["y"] >= g["lo90"]) & (g["y"] <= g["hi90"])
    return {
        "smape": 100 * np.nanmean(smape_terms),
        "mase": err.abs().mean() / scale if scale and scale > 0 else np.nan,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "coverage_80": 100 * in80.mean(),
        "coverage_90": 100 * in90.mean(),
        "width_80": (g["hi80"] - g["lo80"]).mean(),
        "n_points": len(g),
    }


def main():
    sales = pd.read_csv(DATA, parse_dates=["ds"], dtype={"unique_id": str})
    cv = read_all("cv_*.csv")
    cv["ds"] = pd.to_datetime(cv["ds"])
    cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv = cv.dropna(subset=["y"])

    # --- Verifications de coherence entre modeles ---
    cuts_by_model = cv.groupby("model")["cutoff"].apply(lambda s: tuple(sorted(s.unique())))
    if cuts_by_model.nunique() > 1:
        print("ATTENTION : les modeles n'ont pas les memes dates de coupure ; la comparaison est faussee.")
        print(cuts_by_model)
    ids_by_model = cv.groupby("model")["unique_id"].apply(lambda s: tuple(sorted(s.unique())))
    if ids_by_model.nunique() > 1:
        print("ATTENTION : les modeles n'ont pas ete evalues sur les memes series "
              "(un script lance avec --limit ?). Relancez-les tous sur les 15 series.")
        print(ids_by_model.apply(len).rename("nb_series"))

    first_cut = cv["cutoff"].min()
    scales = mase_scales(sales, first_cut)
    test_weeks = sorted(cv["ds"].unique())
    print(f"Periode de test : {pd.Timestamp(test_weeks[0]).date()} -> {pd.Timestamp(test_weeks[-1]).date()} "
          f"({len(test_weeks)} semaines)")

    if HOLIDAYS.exists():
        hol = pd.read_csv(HOLIDAYS, parse_dates=["ds"])
        in_test = hol[hol["ds"].isin(test_weeks)]
        print("Fetes presentes dans la periode de test :",
              [f"{r.holiday} {r.ds.date()}" for r in in_test.itertuples()] or "aucune")

    # --- Metriques par (serie, modele) ---
    rows = []
    for (uid, model), g in cv.groupby(["unique_id", "model"]):
        rows.append({"unique_id": uid, "model": model, **series_metrics(g, scales.get(uid))})
    met = pd.DataFrame(rows)
    zero_scale = [u for u, s in scales.items() if not s or s == 0]
    if zero_scale:
        print("ATTENTION : MASE impossible (echelle nulle) pour :", zero_scale)

    # Meilleur modele par serie : MASE le plus bas, puis sMAPE en cas d'egalite
    met["is_best"] = False
    order = met.sort_values(["unique_id", "mase", "smape"], na_position="last")
    met.loc[order.groupby("unique_id").head(1).index, "is_best"] = True
    met = met.round({"smape": 2, "mase": 3, "rmse": 1, "coverage_80": 1, "coverage_90": 1, "width_80": 1})
    met = met[["unique_id", "model", "smape", "mase", "rmse", "coverage_80", "coverage_90",
               "is_best", "width_80", "n_points"]]
    met.to_csv(OUT / "metrics.csv", index=False)
    print("-> outputs/metrics.csv")

    # --- Resume par modele (diapositive 18) ---
    cv["in80"] = (cv["y"] >= cv["lo80"]) & (cv["y"] <= cv["hi80"])
    cv["in90"] = (cv["y"] >= cv["lo90"]) & (cv["y"] <= cv["hi90"])
    summary = met.groupby("model").agg(
        n_series=("unique_id", "nunique"),
        smape_mean=("smape", "mean"),
        mase_mean=("mase", "mean"),
        mase_median=("mase", "median"),
        rmse_mean=("rmse", "mean"),
        width_80_mean=("width_80", "mean"),
        n_best=("is_best", "sum"),
    )
    # couverture globale = sur TOUS les points de test (pas la moyenne des moyennes)
    summary["coverage_80_global"] = 100 * cv.groupby("model")["in80"].mean()
    summary["coverage_90_global"] = 100 * cv.groupby("model")["in90"].mean()
    summary = summary.round(2).sort_values("mase_mean").reset_index()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    print("-> outputs/metrics_summary.csv")
    print(summary.to_string(index=False))

    timing = OUT / "timing.csv"
    if timing.exists():
        print("\nTemps de calcul (outputs/timing.csv) :")
        print(pd.read_csv(timing).to_string(index=False))

    # --- Previsions futures au format du contrat (pour P3) ---
    fc = read_all("forecast_*.csv")
    fc["p05"], fc["p10"], fc["p50"] = fc["lo90"], fc["lo80"], fc["yhat"]
    fc["p90"], fc["p95"] = fc["hi80"], fc["hi90"]
    q = ["p05", "p10", "p50", "p90", "p95"]
    fc[q] = fc[q].clip(lower=0).round(1)  # une demande ne peut pas etre negative
    bad = ~((fc["p05"] <= fc["p10"]) & (fc["p10"] <= fc["p50"])
            & (fc["p50"] <= fc["p90"]) & (fc["p90"] <= fc["p95"]))
    if bad.any():
        print(f"ATTENTION : {bad.sum()} lignes ou les quantiles ne sont pas dans l'ordre P05<=P10<=P50<=P90<=P95")
    fc["run_date"] = sales["ds"].max().date()  # on se place le dernier vendredi connu
    fc = fc.merge(met[["unique_id", "model", "is_best"]], on=["unique_id", "model"], how="left")
    fc["is_best"] = fc["is_best"].fillna(False).astype(bool)
    fc = fc[["unique_id", "ds", "model"] + q + ["run_date", "is_best"]]
    fc.to_csv(OUT / "forecasts.csv", index=False)
    print(f"-> outputs/forecasts.csv ({fc['unique_id'].nunique()} series, "
          f"semaines {fc['ds'].min()} -> {fc['ds'].max()})")

    if HOLIDAYS.exists():
        fut = pd.to_datetime(fc["ds"]).unique()
        in_fut = hol[hol["ds"].isin(fut)]
        print("Fetes presentes dans les semaines prevues :",
              [f"{r.holiday} {r.ds.date()}" for r in in_fut.itertuples()] or "aucune")


if __name__ == "__main__":
    main()
