# main.py - Projet IA Prédiction Diabète (Version Experte Finale)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
import shap

# Créer le dossier models
os.makedirs("models", exist_ok=True)

# 1. Chargement des données
print("Chargement du dataset...")
df = pd.read_csv('data/diabetes_UCI.csv')

print("\nAperçu des données :")
print(df.head())
print("\nInformations :")
print(df.info())

# 2. Imputation intelligente (0 = valeur manquante)
na_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
df[na_cols] = imputer.fit_transform(df[na_cols])

# 3. Analyse exploratoire
plt.figure(figsize=(8, 5))
sns.countplot(x='Outcome', data=df)
plt.title('Répartition des classes (0 = Pas de diabète, 1 = Diabète)')
plt.show()

print("\nStatistiques descriptives :")
print(df.describe().T)

# 4. Préparation
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# 5. SMOTE
print(f"\nAvant SMOTE : {dict(pd.Series(y_train).value_counts())}")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Après SMOTE : {dict(pd.Series(y_train_res).value_counts())}")

# 6. Modèles
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(random_state=42)
}

plt.figure(figsize=(10, 7))
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*20} {name} (avec SMOTE) {'='*20}")
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f"Matrice de confusion - {name}")
    plt.show()

    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend()
plt.grid(True)
plt.show()

# 7. Optimisation Random Forest
print("\nOptimisation RandomForest...")
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [None, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("Meilleurs paramètres :", grid.best_params_)
best_rf = grid.best_estimator_

# 8. Importance des variables
plt.figure(figsize=(10, 6))
feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance des variables (Random Forest)')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 9. SHAP – VERSION 100% CORRIGÉE ET ROBUSTE
print("\nGénération des graphiques SHAP...")
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# Gestion automatique du format de sortie
if isinstance(shap_values, list):
    shap_pos = shap_values[1]  # classe positive
else:
    shap_pos = shap_values

X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Summary plot classique
shap.summary_plot(shap_pos, X_test_df, show=False)
plt.title("SHAP Summary Plot - Impact sur le risque de diabète")
plt.tight_layout()
plt.show()

# Beeswarm plot (le plus beau)
shap.summary_plot(shap_pos, X_test_df, plot_type="dot", show=False)
plt.title("SHAP Beeswarm Plot")
plt.tight_layout()
plt.show()

# 10. Validation croisée finale
print("\nValidation croisée stratifiée...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
for train_idx, val_idx in skf.split(X_scaled, y):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    best_rf.fit(X_tr_res, y_tr_res)
    auc = roc_auc_score(y_val, best_rf.predict_proba(X_val)[:, 1])
    auc_scores.append(auc)

print(f"AUC moyen : {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}")

# 11. Sauvegarde finale
joblib.dump(best_rf, 'models/random_forest_diabetes_final.pkl')
joblib.dump(scaler, 'models/scaler_diabetes.pkl')
joblib.dump(imputer, 'models/imputer_diabetes.pkl')

print("\nTout est terminé ! Modèles sauvegardés dans le dossier 'models'")
print("Tu peux maintenant lancer : streamlit run app.py")