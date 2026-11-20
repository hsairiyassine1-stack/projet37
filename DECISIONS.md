# DECISIONS.md - Projet 37 (BI2 - G9)

Yassine Hsairi, Abderrahmen Ben Naya, Nedhir Souissi - mis a jour le 10/11/2026

| # | Decision | Choix | Statut |
|---|---|---|---|
| D1 | Titre exact sur la couverture | [titre du PDF ou de la feuille d'inscription - e-mail au prof envoye ? oui/non] | A decider |
| D2 | Langue des diapositives / de l'oral | [anglais / francais] | A decider |
| D3 | Donnees | Kaggle "Walmart Recruiting - Store Sales Forecasting", train.csv (fev. 2010 - oct. 2012) | Decide |
| D4 | Perimetre | 15 series completes tirees au hasard (random_state=42), unique_id = Store_Dept, freq = W-FRI, h = 4 semaines, delai L = 2 semaines | Decide |
| D5 | Niveaux d'intervalle | level = [80, 90] : P10/P90 = bornes a 80 %, P05/P95 = bornes a 90 % | Decide |
| D6 | Canal d'alerte n8n | [e-mail / Telegram / Slack] | A decider |
| D7 | Roles | P1 = [nom], P2 = [nom], P3 = [nom] | A decider |

## Choix techniques
- Validation walk-forward : 5 fenetres de 4 semaines (test du 15/06/2012 au 26/10/2012), calibration conforme sur 4 fenetres de 4 semaines.
- Modeles : SeasonalNaive (reference), AutoARIMA, AutoETS, AutoTheta (StatsForecast, saison = 52) ; Prophet avec les 4 fetes et 4 regresseurs (Temperature, Fuel_Price, CPI, Unemployment).
- Regresseurs dans le futur : derniere valeur connue (pas de valeurs futures reelles : sinon fuite de donnees).
- MarkDown1-5 ecartees (disponibles seulement apres nov. 2011, souvent NA).
- Valeurs negatives (retours) ramenees a 0, tracees dans outputs/anomalies.csv. Les pics de fetes ne sont jamais supprimes.
- Stocks et delais : simules (data/inventory.csv), signale dans la presentation.
- Meilleur modele par serie : MASE le plus bas, puis sMAPE.

## Limites connues (a dire dans la conclusion)
- La periode de test (juin - oct. 2012) ne contient ni Thanksgiving ni Noel : les pics ne sont pas evalues par le backtest.
- AutoETS (methode heritee de R) ne gere pas une saison superieure a 24 : avec 52, sa saisonnalite serait ignoree. [A confirmer par l'avertissement affiche dans la console]
- Intervalles conformes : chaque intervalle repose sur peu d'erreurs passees (4 fenetres) ; la couverture est verifiee, pas garantie.
