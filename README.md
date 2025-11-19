# 🩺 Prédiction du Diabète - Dashboard IA

Ce projet est une application web interactive construite avec **Python** et **Streamlit**. Elle utilise un modèle de Machine Learning (**Random Forest**) pour prédire la probabilité de diabète chez un patient en fonction de données cliniques.

L'application met l'accent sur l'**explicabilité** (XAI) et l'aide à la décision médicale.

## 🚀 Fonctionnalités Clés

* **🤖 Modèle Performant** : Utilisation d'un algorithme *Random Forest* optimisé.
* **⚖️ Gestion du Déséquilibre** : Entraînement réalisé avec **SMOTE** (Synthetic Minority Over-sampling Technique) pour améliorer la détection des cas positifs.
* **🔍 Explicabilité (SHAP)** : Intégration de graphiques *SHAP (Waterfall plot)* pour expliquer pourquoi le modèle a pris telle décision (quel facteur a le plus pesé).
* **📄 Rapport PDF** : Génération automatique d'un rapport médical téléchargeable incluant le diagnostic, les données et le graphique d'analyse.
* **💡 Conseils Personnalisés** : Système de règles métiers fournissant des recommandations de santé basées sur les valeurs critiques (IMC, Glucose, etc.).

## 🛠️ Technologies Utilisées

* **Langage** : Python 3.9+
* **Interface** : Streamlit
* **Machine Learning** : Scikit-learn, Imbalanced-learn (SMOTE)
* **Interprétabilité** : SHAP
* **Manipulation de données** : Pandas, NumPy
* **Visualisation** : Matplotlib
* **Génération de PDF** : FPDF

## 📂 Structure du Projet

```text
├── app.py               # Application principale (Streamlit)
├── requirements.txt     # Liste des dépendances
├── models/              # Dossier contenant les modèles entraînés
│   ├── random_forest_diabetes_final.pkl
│   ├── scaler_diabetes.pkl
│   └── imputer_diabetes.pkl
└── README.md            # Documentation du projet
