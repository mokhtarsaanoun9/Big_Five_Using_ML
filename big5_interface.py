import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# 🧼 Fonction de nettoyage du texte
def clean_text(text):
    import re
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # Supprimer la ponctuation
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 🎨 Config
st.set_page_config(page_title="Big Five Personality Dashboard", layout="wide")
st.markdown("<h1 style='color:#1E90FF;'>🧠 Analyse des traits de personnalité Big Five</h1>", unsafe_allow_html=True)
st.markdown("**Explorez et filtrez les prédictions des traits O, C, E, A, N à partir des avis clients.**")

# 📦 Charger le modèle
model_path = r"C:\taysir\best_modell_.pkl"
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# 📂 Chargement du fichier principal
df_main = pd.read_csv(r"C:\taysir\big_5.csv")
df_main = df_main[["Review Text", "O", "C", "E", "A", "N"]].dropna(subset=["Review Text"]).copy()
df_main["Cleaned Review"] = df_main["Review Text"].apply(clean_text)

# 🔠 Recalcule TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X_main = tfidf.fit_transform(df_main["Cleaned Review"])

# ⏫ Upload utilisateur
st.sidebar.header("📂 Charger un autre fichier à prédire")
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV avec une colonne 'Review Text'", type=["csv"])

if uploaded_file is not None:
    try:
        df_new = pd.read_csv(uploaded_file)
        if "Review Text" not in df_new.columns:
            st.error("❌ Le fichier doit contenir une colonne 'Review Text'.")
        else:
            df_new["Cleaned Review"] = df_new["Review Text"].apply(clean_text)
            X_new = tfidf.transform(df_new["Cleaned Review"])
            y_pred = model.predict(X_new)
            df_pred = pd.DataFrame(y_pred, columns=["O", "C", "E", "A", "N"])
            df_new = pd.concat([df_new, df_pred], axis=1)
            df_main = pd.concat([df_main, df_new], ignore_index=True)
            st.success("✅ Prédictions ajoutées avec succès au tableau !")
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier : {e}")

# 🎛️ Filtres
st.sidebar.header("⚙️ Options de filtrage")
keyword = st.sidebar.text_input("🔍 Rechercher un mot dans les commentaires")
show_avg = st.sidebar.checkbox("📊 Afficher les moyennes par trait")
num_rows = st.sidebar.slider("📌 Nombre de commentaires à afficher", min_value=5, max_value=200, value=20)

# 🔍 Filtrage
df_main[["O", "C", "E", "A", "N"]] = df_main[["O", "C", "E", "A", "N"]].round(2)
df_filtered = df_main[df_main["Review Text"].str.contains(keyword, case=False, na=False)] if keyword else df_main

# 📘 Légende
st.markdown("""
> ℹ️ **O** : Ouverture à l'expérience &nbsp;&nbsp;&nbsp;&nbsp; **C** : Conscienciosité  
> **E** : Extraversion &nbsp;&nbsp;&nbsp;&nbsp; **A** : Agréabilité &nbsp;&nbsp;&nbsp;&nbsp; **N** : Névrosisme
""")

# 📊 Moyennes
if show_avg:
    st.subheader("📈 Moyenne des traits Big Five")
    st.bar_chart(df_filtered[["O", "C", "E", "A", "N"]].mean())

# 📋 Tableau stylisé
st.markdown("### 📝 Tableau structuré des prédictions Big Five")
st.dataframe(
    df_filtered.head(num_rows).style.format({
        "O": "{:.2f}", "C": "{:.2f}", "E": "{:.2f}", "A": "{:.2f}", "N": "{:.2f}"
    }).background_gradient(cmap="Blues", subset=["O", "C", "E", "A", "N"]),
    use_container_width=True
)

# 📥 Télécharger
st.download_button("⬇️ Télécharger les résultats filtrés", df_filtered.to_csv(index=False), "bigfive_results.csv", "text/csv")

# ✨ Footer
st.markdown("""
    <hr style='border:1px solid #ddd'>
    <p style='text-align:center; color:gray'>
    Réalisé avec ❤️ par <b>Tayssir Taam</b> — 2025<br>
    Dashboard Big Five Personality • NLP & ML
    </p>
""", unsafe_allow_html=True)
