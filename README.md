# FakeMoneyTrack 🛡️
**Détection de Faux Billets par Analyse de Texture**  
INF4238 — Vision par Ordinateur · Groupe 6 · Master 1 Data Science & IA · UY1

---

## Structure du projet

```
fakemoneytrack/
├── server.py              # Backend Flask (API REST)
├── train_pipeline.py      # Entraînement SVM + génération graphiques
├── templates/
│   └── index.html         # Frontend HTML/CSS/JS (interface principale)
├── requirements.txt
├── Procfile               # Pour Render
├── Dockerfile
└── data_banknote_authentication.txt  # Auto-téléchargé au premier run
```

## Lancement local

```bash
# 1. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Entraîner le modèle (génère model_svm_banknote.pkl)
python3 train_pipeline.py

# 4. Lancer le serveur Flask
python3 server.py

# L'app est accessible sur http://localhost:5000
```

## Déploiement sur Render

1. Pousser ce dossier sur GitHub
2. Sur render.com → New Web Service → connecter le repo
3. Build command : `pip install -r requirements.txt && python3 train_pipeline.py`
4. Start command : `gunicorn server:app --bind 0.0.0.0:$PORT`

## Fonctionnalités

- **Import d'image** — glisser-déposer ou parcourir, extraction automatique des descripteurs par ondelettes de Haar
- **Caméra webcam** — capture temps réel avec cadre de placement du billet
- **Saisie manuelle** — entrée directe des 4 descripteurs (mode expert)
- **Résultats détaillés** — probabilités, barres de confiance, visualisation des 4 features
- **Historique persistant** — stockage localStorage des 50 dernières analyses
- **Design sombre moderne** — UI professionnelle responsive

## Pipeline technique

1. CLAHE + redimensionnement 256×256
2. DWT2 (ondelettes de Haar) → coefficients d'approximation
3. Extraction : Variance, Skewness, Kurtosis, Entropie
4. SVM RBF (C=10, gamma=scale, Platt scaling)
5. Décision binaire + probabilités de confiance

## Résultats modèle

| Métrique  | Authentique | Faux   | Global |
|-----------|------------|--------|--------|
| Précision | 1.00       | 1.00   | 1.00   |
| Rappel    | 1.00       | 1.00   | 1.00   |
| F1-score  | 1.00       | 1.00   | 1.00   |

AUC-ROC : **1.000** · Dataset : UCI Banknote Authentication (1372 échantillons)

---
*Université de Yaoundé 1 · Département d'Informatique · 2025–2026*
