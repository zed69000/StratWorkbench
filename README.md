# StratWorkbench

Outil de backtest et d'optimisation de stratégies trading en Python. Interface Streamlit pour charger des séries, tester des stratégies, appliquer des filtres et lancer des optimisations.

## Fonctions clés
- Chargement de séries CSV ou génération synthétique.  
- Découverte dynamique de stratégies et filtres.  
- Backtest réaliste avec coûts, slippage, stop-loss et métriques (Sharpe, drawdown, indices).  
- Optimiseur par grille retournant `best_params.json`. :contentReference[oaicite:0]{index=0}

## Structure du dépôt
- `app.py` : interface Streamlit et orchestration.  
- `core/` : moteur, découverte de stratégies, backtest et métriques.  
- `strats/` : stratégies (generate_signals).  
- `filter/` : filtres applicables aux signaux.  
- `__extUtils`, `utils`, `requirements.txt`, `run.ps1`, `best_params.json`. :contentReference[oaicite:1]{index=1}

## Quickstart
1. Créer un environnement Python 3.10+.  
2. `pip install -r requirements.txt`.  
3. Lancer l'UI : `streamlit run app.py`. :contentReference[oaicite:2]{index=2}

## Workflow résumé
1. Charger ou générer les données.  
2. Sélectionner stratégies et filtres dans l'UI.  
3. Exécuter backtest.  
4. Lancer l'optimisation pour chercher les meilleurs paramètres. :contentReference[oaicite:3]{index=3}

## Améliorations possibles
- Rendre l'optimiseur moins sujet à overfitting (heuristiques, réduction d'espace).  
- Ajuster `core/metrics.py` pour coller aux coûts réels. :contentReference[oaicite:4]{index=4}

## Contribution
Fork, push et ouvre une Pull Request. Pour questions ouvre une Issue.
