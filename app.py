# app.py – Dashboard Prédiction Diabète (VERSION FINALE : PDF COMPLET AVEC CONSEILS)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Configuration
st.set_page_config(page_title="Prédiction Diabète", page_icon="🏥", layout="centered")

st.title("🩺 Prédiction du Diabète")
st.markdown("### Modèle Random Forest optimisé + SMOTE + Explicabilité SHAP")

# ---------------------------------------------------------
# 1. GÉNÉRATION DES CONSEILS (Logique extraite)
# ---------------------------------------------------------
def generer_liste_conseils(input_df, prediction_classe):
    """Génère une liste de conseils sous forme de texte."""
    conseils = []
    
    # Récupération des valeurs
    glucose = input_df['Glucose'].values[0]
    bmi = input_df['BMI'].values[0]
    bp = input_df['BloodPressure'].values[0]
    age = input_df['Age'].values[0]
    
    # Logique des conseils
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

# ---------------------------------------------------------
# 2. FONCTION DE GÉNÉRATION PDF (Mise à jour)
# ---------------------------------------------------------
def creer_pdf(input_data, prediction_proba, resultat_texte, shap_plot_filename, liste_conseils):
    pdf = FPDF()
    pdf.add_page()

    # --- Titre ---
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 20, "Rapport de Prédiction du Diabète", ln=True, align='C')
    pdf.ln(10)

    # --- Section 1 : Résultat ---
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Résultat de l'analyse", ln=True)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Probabilité détectée : {prediction_proba:.1f}%", ln=True)
    
    if prediction_proba > 50:
        pdf.set_text_color(192, 57, 43) # Rouge
    else:
        pdf.set_text_color(39, 174, 96) # Vert
        
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Diagnostic estimé : {resultat_texte}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # --- Section 2 : Données Patient ---
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

    # --- Section 3 : Conseils (NOUVEAU) ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. Recommandations de Santé", ln=True)
    pdf.set_font("Arial", "", 11)
    
    for conseil in liste_conseils:
        # Nettoyage pour encodage PDF
        txt_conseil = "- " + conseil.encode('latin-1', 'replace').decode('latin-1')
        # Multi_cell permet le retour à la ligne automatique si le texte est long
        pdf.multi_cell(0, 7, txt_conseil)
        pdf.ln(1)

    pdf.ln(5)

    # --- Section 4 : Image SHAP ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "4. Facteurs d'influence (Graphique SHAP)", ln=True)
    pdf.ln(5)
    
    if os.path.exists(shap_plot_filename):
        pdf.image(shap_plot_filename, x=10, w=190)

    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------
# CHARGEMENT ET SAISIE
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("models/random_forest_diabetes_final.pkl")
    scaler = joblib.load("models/scaler_diabetes.pkl")
    imputer = joblib.load("models/imputer_diabetes.pkl")
    return model, scaler, imputer

model, scaler, imputer = load_models()

st.sidebar.header("📋 Informations du patient")
def get_input():
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
    return pd.DataFrame(data, index=[0])

input_df = get_input()

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

# ========================================
# AFFICHAGE DES CONSEILS (ÉCRAN)
# ========================================
st.markdown("---")
st.subheader("💡 Conseils de Santé Personnalisés")

# On génère la liste des conseils ici
mes_conseils = generer_liste_conseils(input_clean, pred)

# On les affiche à l'écran
for conseil in mes_conseils:
    st.info(conseil)

# ========================================
# SHAP EXPLAINER
# ========================================
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

# Waterfall Plot + Sauvegarde
fig_shap = plt.figure(figsize=(10, 6))
shap.plots.waterfall(explanation, show=False)
plt.title("Waterfall Plot SHAP")
plt.tight_layout()
st.pyplot(fig_shap)
fig_shap.savefig("shap_temp.png", bbox_inches='tight', dpi=150)
plt.close(fig_shap)

# ========================================
# PDF EXPORT
# ========================================
st.markdown("---")
st.header("📂 Exportation")

dict_patient = input_df.iloc[0].to_dict()

if st.button("📄 Générer le Rapport PDF"):
    # On passe 'mes_conseils' à la fonction PDF
    pdf_bytes = creer_pdf(dict_patient, proba, resultat_texte_pdf, "shap_temp.png", mes_conseils)
    
    st.success("Rapport généré !")
    st.download_button(label="📥 Télécharger le Rapport Complet", data=pdf_bytes, file_name="rapport_medical_complet.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Projet IA Diabète 2025")