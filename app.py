"""
Heart Disease Predictor — Application Streamlit autonome
Les modèles sont chargés directement (pas d'API externe).

Lancement :
    streamlit run app.py

Prérequis :
    pip install streamlit scikit-learn pandas numpy joblib matplotlib plotly
    → Le dossier models_export/ doit être au même niveau que ce fichier.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st


# CONFIG PAGE

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS PERSONNALISÉ

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.95rem; }

/* Cards */
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: .8rem;
}

/* Result banners */
.result-sick {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-healthy {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.result-title { font-size: 1.6rem; font-weight: 700; margin: 0 0 .3rem; }
.result-sub   { font-size: 0.9rem; opacity: .75; margin: 0; }

/* Metric pills */
.metric-pill {
    display: inline-block;
    background: #f3f4f6;
    border-radius: 999px;
    padding: .3rem .9rem;
    font-size: .82rem;
    font-weight: 500;
    color: #374151;
    margin: .2rem;
}
.metric-pill span { color: #4f46e5; font-weight: 700; }

/* Feature importance bars */
.feat-bar-bg {
    background: #f3f4f6;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
}
.feat-bar-fill {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    height: 100%;
    border-radius: 999px;
}

/* Section titles */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1f2937;
    margin: 1.2rem 0 .7rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



# CHARGEMENT DES MODÈLES (directement, sans API)

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "models_export")

MODEL_DISPLAY = {
    "logistic_regression": "Régression Logistique",
    "random_forest":       "Random Forest",
    "knn":                 "K-Nearest Neighbors",
}

MODEL_COLORS = {
    "logistic_regression": "#a78bfa",
    "random_forest":       "#34d399",
    "knn":                 "#f97316",
}

FEATURE_LABELS = {
    "age":      "Âge",
    "sex":      "Sexe",
    "cp":       "Douleur thoracique",
    "trestbps": "Pression artérielle",
    "chol":     "Cholestérol",
    "fbs":      "Glycémie > 120",
    "restecg":  "ECG au repos",
    "thalach":  "FC max",
    "exang":    "Angine à l'effort",
    "oldpeak":  "Oldpeak",
    "slope":    "Pente ST",
    "ca":       "Vaisseaux colorés",
    "thal":     "Thalassémie",
}


@st.cache_resource(show_spinner="Chargement des modèles ML…")
def load_models():
    """Charge les modèles, le scaler et les métadonnées depuis models_export/."""
    if not os.path.isdir(EXPORT_DIR):
        return None, None, None

    try:
        with open(os.path.join(EXPORT_DIR, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)

        scaler = joblib.load(os.path.join(EXPORT_DIR, "scaler.pkl"))

        ml_models = {}
        for key in meta["models"]:
            path = os.path.join(EXPORT_DIR, f"{key}.pkl")
            if os.path.exists(path):
                ml_models[key] = joblib.load(path)

        return ml_models, scaler, meta
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None, None, None


def predict(ml_models, scaler, meta, patient_values: dict,
            model_key: str, threshold: float = 0.5):
    """Effectue une prédiction avec le modèle choisi."""
    features = meta["features"]
    x = np.array([patient_values[f] for f in features], dtype=float).reshape(1, -1)

    if meta["models"][model_key]["needs_scaling"]:
        x = scaler.transform(x)

    proba = ml_models[model_key].predict_proba(x)[0]
    pred  = int(proba[1] >= threshold)

    return {
        "prediction":    pred,
        "label":         meta["classes"][str(pred)],
        "confidence":    round(float(proba[pred]) * 100, 2),
        "proba_sain":    round(float(proba[0]) * 100, 2),
        "proba_malade":  round(float(proba[1]) * 100, 2),
        "model_key":     model_key,
        "threshold":     threshold,
    }


def get_feature_importance(ml_models, meta, model_key):
    """Retourne les importances de features si disponibles."""
    m = ml_models.get(model_key)
    if m is None:
        return None
    if hasattr(m, "feature_importances_"):          # Random Forest, trees
        imps = m.feature_importances_
    elif hasattr(m, "coef_"):                        # Logistic Regression
        imps = np.abs(m.coef_[0])
        imps = imps / imps.sum()
    else:
        return None
    return dict(zip(meta["features"], imps.tolist()))



# CHARGEMENT

ml_models, scaler, meta = load_models()

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div style="font-size:3rem;line-height:1">❤️</div>
  <div>
    <h1>Détection de maladie cardiaque</h1>
    <p>Prédiction par apprentissage automatique · Logistic Regression · Random Forest · KNN</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Vérification des modèles ──────────────────────────────────────────
if ml_models is None or len(ml_models) == 0:
    st.error(
        "**Dossier `models_export/` introuvable ou vide.**\n\n"
        "Générez les modèles en exécutant le notebook `heart.ipynb` jusqu'à la "
        "section *Exportation de modèle*, puis relancez l'application."
    )
    st.code("streamlit run app.py", language="bash")
    st.stop()

available_models = list(ml_models.keys())
best_model       = meta.get("best_model", available_models[0])

# MISE EN PAGE PRINCIPALE

left_col, right_col = st.columns([1, 1], gap="large")


# COLONNE GAUCHE — Paramètres
with left_col:
    st.markdown('<div class="section-title"> Configuration du modèle</div>',
                unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Modèle ML",
        options=available_models,
        index=available_models.index(best_model) if best_model in available_models else 0,
        format_func=lambda k: f"{'★ ' if k == best_model else ''}{MODEL_DISPLAY.get(k, k)}",
    )

    threshold = st.slider("Seuil de décision", 0.10, 0.90, 0.50, 0.05,
                          help="Seuil de probabilité pour classer un patient comme malade.")

    st.markdown("---")
    st.markdown('<div class="section-title">🩺 Paramètres cliniques</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age  = st.slider("Âge", 29, 77, 55)
        sex  = st.radio("Sexe", [0, 1],
                        format_func=lambda x: "Femme" if x == 0 else "Homme",
                        horizontal=True)
        cp   = st.selectbox("Type de douleur thoracique",
                            [0, 1, 2, 3],
                            format_func=lambda x: [
                                "0 – Angine typique",
                                "1 – Angine atypique",
                                "2 – Non angineux",
                                "3 – Asymptomatique"
                            ][x])
        trestbps = st.slider("Pression artérielle repos (mmHg)", 90, 200, 130)
        chol     = st.slider("Cholestérol (mg/dl)", 100, 570, 240)
        fbs      = st.radio("Glycémie à jeun > 120 mg/dl", [0, 1],
                            format_func=lambda x: "Non" if x == 0 else "Oui",
                            horizontal=True)
        restecg  = st.selectbox("ECG au repos",
                                [0, 1, 2],
                                format_func=lambda x: [
                                    "0 – Normal",
                                    "1 – Anomalie onde ST-T",
                                    "2 – HVG probable"
                                ][x])

    with c2:
        thalach = st.slider("FC maximale atteinte (bpm)", 70, 210, 150)
        exang   = st.radio("Angine induite par l'effort", [0, 1],
                           format_func=lambda x: "Non" if x == 0 else "Oui",
                           horizontal=True)
        oldpeak = st.slider("Dépression ST (oldpeak)", 0.0, 6.0, 1.0, 0.1)
        slope   = st.selectbox("Pente du segment ST",
                               [0, 1, 2],
                               format_func=lambda x: [
                                   "0 – Descendante",
                                   "1 – Plate",
                                   "2 – Ascendante"
                               ][x])
        ca      = st.selectbox("Vaisseaux colorés (fluoroscopie)", [0, 1, 2, 3, 4])
        thal    = st.selectbox("Thalassémie",
                               [0, 1, 2, 3],
                               format_func=lambda x: [
                                   "0 – Normal",
                                   "1 – Défect fixe",
                                   "2 – Normal (flux ok)",
                                   "3 – Défect réversible"
                               ][x])

    predict_btn = st.button("Lancer la prédiction", type="primary",
                            use_container_width=True)

# COLONNE DROITE — Résultats
with right_col:

    #  Performances du modèle sélectionné 
    st.markdown('<div class="section-title">Performances du modèle sélectionné</div>',
                unsafe_allow_html=True)

    m_info = meta["models"].get(model_choice, {})
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy",  f"{m_info.get('accuracy', 0):.1f}%")
    mc2.metric("F1-Score",  f"{m_info.get('f1_score', 0):.1f}%")
    mc3.metric("AUC-ROC",   f"{m_info.get('auc_roc', 0):.1f}%")
    mc4.metric("CV 5-fold", f"{m_info.get('cv_mean', 0):.1f}%")

    st.markdown("---")

    #  Résultat de la prédiction 
    if predict_btn:
        patient_values = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal,
        }

        result = predict(ml_models, scaler, meta, patient_values,
                         model_choice, threshold)

        # Bandeau résultat
        if result["prediction"] == 1:
            st.markdown(f"""
            <div class="result-sick">
              <p class="result-title" style="color:#dc2626"> Maladie cardiaque détectée</p>
              <p class="result-sub">Confiance : <strong>{result['confidence']:.1f}%</strong>
                 · Seuil appliqué : {threshold}</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
              <p class="result-title" style="color:#16a34a"> Patient classé Sain</p>
              <p class="result-sub">Confiance : <strong>{result['confidence']:.1f}%</strong>
                 · Seuil appliqué : {threshold}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        p1, p2 = st.columns(2)
        p1.metric("Probabilité Sain",   f"{result['proba_sain']:.1f}%")
        p2.metric("Probabilité Malade", f"{result['proba_malade']:.1f}%")

        # Caption modèle
        st.caption(
            f"Modèle : **{MODEL_DISPLAY.get(model_choice, model_choice)}** "
            f"· Seuil : **{threshold}** "
            f"· Features : {len(meta['features'])}"
        )

    else:
        st.info("Renseignez les paramètres cliniques puis cliquez sur **Lancer la prédiction**.")



# SECTION BASSE — Comparaison des modèles
st.markdown("---")
st.markdown('<div class="section-title">Comparaison des modèles chargés</div>',
            unsafe_allow_html=True)

rows = []
for key, info in meta["models"].items():
    if key in ml_models:
        rows.append({
            "Modèle":    MODEL_DISPLAY.get(key, key),
            "Accuracy":  info.get("accuracy", 0),
            "F1-Score":  info.get("f1_score", 0),
            "AUC-ROC":   info.get("auc_roc", 0),
            "CV Moy.":   info.get("cv_mean", 0),
            "Meilleur":  "★" if key == best_model else "",
        })

if rows:
    df_comp = pd.DataFrame(rows).set_index("Modèle")

    # Tableau récapitulatif
    st.dataframe(
        df_comp.style.format("{:.2f}%", subset=["Accuracy", "F1-Score", "AUC-ROC", "CV Moy."])
               .highlight_max(subset=["Accuracy", "F1-Score", "AUC-ROC"],
                              color="#214130")
               .set_properties(**{"text-align": "center"}),
        use_container_width=True,
    )
