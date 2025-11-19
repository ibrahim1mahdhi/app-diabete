# app.py – Dashboard Prédiction Diabète (MULTI-PAGES : PRÉDICTION + PERFORMANCE)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import os

# ---------------------------------------------------------
# 0. CONFIGURATION GLOBALE
# ---------------------------------------------------------
st.set_page_config(page_title="Prédiction Diabète", page_icon="🏥", layout="wide")

# ---------------------------------------------------------
# 1. FONCTIONS UTILITAIRES (PDF & CONSEILS)
# ---------------------------------------------------------
def generer_liste_conseils(input_df, prediction_classe):
    """Génère une liste de conseils sous forme de texte."""
    conseils = []
    glucose = input_df['Glucose'].values[0]
    bmi = input_df['BMI'].values[0]
    bp = input_df['BloodPressure'].values[0]
    age = input_df['Age'].values[0]
    
    if prediction_classe == 1:
        conseils.append("Avis global : Compte tenu du risque élevé, une consultation médicale est vivement recommandée.")
    if glucose > 140:
        conseils.append("Glycémie élevée : Réduisez l'apport en sucres rapides et privilégiez les aliments à index glycémique bas.")
    if bmi > 30:
        conseils.append("IMC (Obésité) : Une activité physique régulière (30 min de marche/jour) aide à réduire la résistance à l'insuline.")
    elif bmi > 25:
        conseils.append("IMC (Surpoids) : Surveillez votre poids, même une perte légère (5-10%) réduit les risques.")
    if bp > 80:
        conseils.append("Tension artérielle : Limitez la consommation de sel et d'alcool. Gérez votre stress.")
    if age > 45 and bmi > 25:
        conseils.append("Age + Poids : Après 45 ans, le métabolisme change. Un bilan sanguin annuel est conseillé.")
    if not conseils:
        conseils.append("Vos indicateurs principaux (Glucose, IMC, Tension) sont dans la moyenne. Continuez ainsi !")
    return conseils

def creer_pdf(input_data, prediction_proba, resultat_texte, shap_plot_filename, liste_conseils):
    pdf = FPDF()
    pdf.add_page()

    # Titre
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 20, "Rapport de Prédiction du Diabète", ln=True, align='C')
    pdf.ln(10)

    # Section 1
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Résultat de l'analyse", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Probabilité détectée : {prediction_proba:.1f}%", ln=True)
    if prediction_proba > 50:
        pdf.set_text_color(192, 57, 43)
    else:
        pdf.set_text_color(39, 174, 96)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Diagnostic estimé : {resultat_texte}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Section 2
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Données cliniques saisies", ln=True)
    pdf.set_font("Arial", "", 11)
    col_width = 90
    row_height = 7
    for variable, valeur in input_data.items():
        try:
            var_str = str(variable).encode('latin-1', 'replace').decode('latin-1')
            val_str = str(valeur).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(col_width, row_height, var_str, border=1)
            pdf.cell(col_width, row_height, val_str, border=1, ln=True)
        except:
            pass
    pdf.ln(5)

    # Section 3
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. Recommandations de Santé", ln=True)
    pdf.set_font("Arial", "", 11)
    for conseil in liste_conseils:
        txt_conseil = "- " + conseil.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, txt_conseil)
        pdf.ln(1)
    pdf.ln(5)

    # Section 4
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "4. Facteurs d'influence (Graphique SHAP)", ln=True)
    pdf.ln(5)
    if os.path.exists(shap_plot_filename):
        pdf.image(shap_plot_filename, x=10, w=190)
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------
# 2. CHARGEMENT DES RESSOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("models/random_forest_diabetes_final.pkl")
    scaler = joblib.load("models/scaler_diabetes.pkl")
    imputer = joblib.load("models/imputer_diabetes.pkl")
    return model, scaler, imputer

model, scaler, imputer = load_models()

@st.cache_data
def load_data():
    # Assurez-vous que le fichier CSV est bien dans le dossier data/ ou à la racine
    if os.path.exists('data/diabetes_UCI.csv'):
        return pd.read_csv('data/diabetes_UCI.csv')
    elif os.path.exists('diabetes_UCI.csv'):
        return pd.read_csv('diabetes_UCI.csv')
    else:
        return None

# ---------------------------------------------------------
# 3. PAGE A : INTERFACE UTILISATEUR (Votre code original)
# ---------------------------------------------------------
def page_prediction():
    st.title("🩺 Prédiction du Diabète")
    st.markdown("### Modèle Random Forest optimisé + SMOTE + Explicabilité SHAP")

    st.sidebar.header("📋 Informations du patient")
    
    # Saisie des données
    preg = st.sidebar.slider("Grossesses", 0, 17, 3)
    gluc = st.sidebar.slider("Glucose (mg/dL)", 0, 200, 120)
    bp = st.sidebar.slider("Pression artérielle (mm Hg)", 0, 122, 72)
    skin = st.sidebar.slider("Épaisseur pli cutané (mm)", 0, 99, 20)
    ins = st.sidebar.slider("Insuline (μU/mL)", 0, 900, 80)
    bmi = st.sidebar.slider("IMC", 0.0, 70.0, 30.0, 0.1)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
    age = st.sidebar.slider("Âge (ans)", 21, 90, 30)

    data = {
        'Pregnancies': preg, 'Glucose': gluc, 'BloodPressure': bp,
        'SkinThickness': skin, 'Insulin': ins, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }
    input_df = pd.DataFrame(data, index=[0])

    st.subheader("🔍 Valeurs saisies")
    st.write(input_df.T)

    # Pré-processing
    na_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_clean = input_df.copy()
    input_clean[na_cols] = imputer.transform(input_clean[na_cols])
    input_scaled = scaler.transform(input_clean)

    # Prédiction
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1] * 100

    st.markdown("---")
    col1, col2 = st.columns(2)
    resultat_texte_pdf = ""
    with col1:
        if pred == 1:
            st.error("⚠️ **RISQUE ÉLEVÉ DE DIABÈTE**")
            resultat_texte_pdf = "RISQUE ELEVE (Positif)"
        else:
            st.success("✅ **FAIBLE RISQUE DE DIABÈTE**")
            resultat_texte_pdf = "FAIBLE RISQUE (Negatif)"

    with col2:
        st.metric("Probabilité de diabète", f"{proba:.1f}%")

    # Jauge
    fig_gauge, ax = plt.subplots(figsize=(10, 0.8))
    color = "red" if proba > 50 else "orange" if proba > 30 else "green"
    ax.barh(0, proba, color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.axis('off')
    st.pyplot(fig_gauge)
    plt.close(fig_gauge)

    # Conseils
    st.markdown("---")
    st.subheader("💡 Conseils de Santé Personnalisés")
    mes_conseils = generer_liste_conseils(input_clean, pred)
    for conseil in mes_conseils:
        st.info(conseil)

    # SHAP
    st.markdown("---")
    st.subheader("🔍 Pourquoi cette prédiction ? (Explicabilité SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    if isinstance(shap_values, list):
        shap_1d = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_1d = shap_values[0, :, 1]
        base_value = explainer.expected_value[1]

    explanation = shap.Explanation(values=shap_1d, base_values=base_value, data=input_scaled[0], feature_names=input_df.columns.tolist())
    
    fig_shap = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    st.pyplot(fig_shap)
    fig_shap.savefig("shap_temp.png", bbox_inches='tight', dpi=150)
    plt.close(fig_shap)

    # PDF
    st.markdown("---")
    st.header("📂 Exportation")
    dict_patient = input_df.iloc[0].to_dict()
    if st.button("📄 Générer le Rapport PDF"):
        pdf_bytes = creer_pdf(dict_patient, proba, resultat_texte_pdf, "shap_temp.png", mes_conseils)
        st.success("Rapport généré !")
        st.download_button(label="📥 Télécharger le Rapport Complet", data=pdf_bytes, file_name="rapport_medical_complet.pdf", mime="application/pdf")

# ---------------------------------------------------------
# 4. PAGE B : DASHBOARD TECHNIQUE (Nouvelle page)
# ---------------------------------------------------------
def page_dashboard():
    st.title("📊 Tableau de Bord Technique (Performance)")
    st.markdown("Analyse de la performance du modèle sur les données de test.")
    
    df = load_data()
    if df is None:
        st.warning("⚠️ Fichier 'data/diabetes_UCI.csv' non trouvé. Impossible d'afficher les métriques.")
        st.info("Veuillez ajouter le fichier CSV dans votre dépôt GitHub.")
        return

    with st.spinner("Calcul des métriques en cours..."):
        # Préparation des données
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Application des mêmes transformations (Imputer/Scaler)
        na_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        X_clean = X.copy()
        X_clean[na_cols] = imputer.transform(X_clean[na_cols])
        X_scaled = scaler.transform(X_clean)
        
        # Split Train/Test pour évaluation honnête
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
    # KPI
    col1, col2, col3 = st.columns(3)
    acc = accuracy_score(y_test, y_pred)
    roc_auc = pd.Series(y_proba).corr(pd.Series(y_test)) # Approximation rapide pour affichage
    
    col1.metric("Précision (Accuracy)", f"{acc:.2%}")
    col2.metric("Données de Test", f"{len(y_test)} patients")
    col3.metric("Algorithme", "Random Forest + SMOTE")
    
    st.divider()
    
    # Graphiques Matrice & ROC
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Sain', 'Diabète'], yticklabels=['Sain', 'Diabète'])
        plt.ylabel('Vraie Classe')
        plt.xlabel('Prédiction')
        st.pyplot(fig_cm)
        
    with col_g2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC (approx)')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taux Faux Positifs')
        ax.set_ylabel('Taux Vrais Positifs')
        st.pyplot(fig_roc)
        
    # Feature Importance
    st.divider()
    st.subheader("Importance des Variables")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    fig_feat, ax = plt.subplots(figsize=(10, 4))
    plt.title("Impact des variables sur la décision du modèle")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    st.pyplot(fig_feat)

# ---------------------------------------------------------
# 5. NAVIGATION (Menu Latéral)
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers :", ["🔮 Prédiction Patient", "📈 Dashboard Technique"])

if page == "🔮 Prédiction Patient":
    page_prediction()
else:
    page_dashboard()

st.sidebar.markdown("---")
st.sidebar.caption("Projet IA Diabète 2025")