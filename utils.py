from fpdf import FPDF
import matplotlib.pyplot as plt
import streamlit as st
import os

def creer_pdf(input_data, prediction_proba, resultat_texte, shap_plot_filename="shap_temp.png"):
    """
    Génère un rapport PDF avec les données, le score et l'image du graphique SHAP.
    """
    pdf = FPDF()
    pdf.add_page()

    # --- Titre ---
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(44, 62, 80) # Bleu foncé
    pdf.cell(0, 20, "Rapport de Prédiction du Diabète", ln=True, align='C')
    pdf.ln(10)

    # --- Section 1 : Résultat ---
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Résultat de l'analyse", ln=True)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Probabilité détectée : {prediction_proba*100:.2f}%", ln=True)
    
    # Couleur conditionnelle pour le texte (Rouge si risque élevé, Vert sinon)
    if prediction_proba > 0.5:
        pdf.set_text_color(192, 57, 43) # Rouge
    else:
        pdf.set_text_color(39, 174, 96) # Vert
        
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Diagnostic estimé : {resultat_texte}", ln=True)
    pdf.set_text_color(0, 0, 0) # Reset couleur
    pdf.ln(10)

    # --- Section 2 : Données du Patient ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Données cliniques saisies", ln=True)
    pdf.set_font("Arial", "", 12)
    
    # Création d'un petit tableau simple
    col_width = 90
    row_height = 8
    
    for variable, valeur in input_data.items():
        pdf.cell(col_width, row_height, f"{variable}", border=1)
        pdf.cell(col_width, row_height, f"{valeur}", border=1, ln=True)
    
    pdf.ln(10)

    # --- Section 3 : Explicabilité (SHAP) ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "3. Facteurs d'influence (Graphique SHAP)", ln=True)
    pdf.ln(5)
    
    # Vérifier si l'image existe avant de l'ajouter
    if os.path.exists(shap_plot_filename):
        # (x, y, w) -> w=190 pour prendre toute la largeur de la page A4 moins les marges
        pdf.image(shap_plot_filename, x=10, w=190)
    else:
        pdf.cell(0, 10, "Graphique non disponible", ln=True)

    # Retourner le contenu du PDF en bytes
    return pdf.output(dest='S').encode('latin-1')