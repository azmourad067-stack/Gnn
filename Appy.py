import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import json
from datetime import datetime
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, precision_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==== CONFIGURATIONS MACHINE LEARNING ====
ML_CONFIG = {
    "model_type": "xgboost",  # "logistic", "random_forest", "xgboost", "gradient_boosting"
    "target_variable": "top3",  # "winner" ou "top3"
    "test_size": 0.2,
    "cross_validation": 5,
    "random_state": 42,
    "feature_importance_threshold": 0.01,
    "calibration": True
}

class HorseRacingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.performance_metrics = {}
        
    def create_synthetic_labels(self, df, method="odds_based"):
        """
        Crée des labels synthétiques pour l'entraînement basés sur:
        - Les cotes (méthode principale)
        - Les patterns de performance historique
        """
        labels = pd.Series(0, index=df.index)
        
        if method == "odds_based":
            # Conversion des cotes en probabilités implicites
            df['implied_prob'] = 1 / df['odds_numeric']
            total_prob = df['implied_prob'].sum()
            
            if total_prob > 0:
                df['normalized_prob'] = df['implied_prob'] / total_prob
                
                # Génération de labels basés sur les probabilités
                np.random.seed(ML_CONFIG["random_state"])
                for idx, row in df.iterrows():
                    if ML_CONFIG["target_variable"] == "winner":
                        # Label binaire pour gagnant
                        if np.random.random() < row['normalized_prob'] * 0.8:  # Facteur de calibration
                            labels.loc[idx] = 1
                    else:  # top3
                        # Label pour top 3 (plus fréquent)
                        top3_prob = min(row['normalized_prob'] * 2.5, 0.95)  # Amplification pour top3
                        if np.random.random() < top3_prob:
                            labels.loc[idx] = 1
                            
        elif method == "composite":
            # Combinaison de multiples facteurs
            pass
            
        return labels

    def engineer_features(self, df):
        """
        Crée des features avancées pour la modélisation
        """
        features_df = df.copy()
        
        # Features de base
        base_features = [
            'odds_numeric', 'draw_numeric', 'weight_kg'
        ]
        
        # Features dérivées des cotes
        features_df['odds_reciprocal'] = 1 / features_df['odds_numeric']
        features_df['odds_log'] = np.log(features_df['odds_numeric'])
        features_df['odds_rank'] = features_df['odds_numeric'].rank()
        
        # Features de position relative
        features_df['draw_position_ratio'] = features_df['draw_numeric'] / features_df['draw_numeric'].max()
        features_df['is_inner_draw'] = (features_df['draw_numeric'] <= 4).astype(int)
        features_df['is_outer_draw'] = (features_df['draw_numeric'] >= features_df['draw_numeric'].max() - 2).astype(int)
        
        # Features de poids
        features_df['weight_deviation'] = (features_df['weight_kg'] - features_df['weight_kg'].mean()) / features_df['weight_kg'].std()
        features_df['is_light_weight'] = (features_df['weight_kg'] < features_df['weight_kg'].quantile(0.3)).astype(int)
        
        # Analyse de la "musique" (performances récentes)
        if 'Musique' in df.columns:
            features_df['recent_perf_score'] = df['Musique'].apply(self._parse_musique_score)
        
        # Features d'interaction
        features_df['odds_draw_interaction'] = features_df['odds_reciprocal'] * (1 / features_df['draw_numeric'])
        features_df['odds_weight_interaction'] = features_df['odds_reciprocal'] * (1 / features_df['weight_kg'])
        
        # Sélection des features finales
        feature_columns = [col for col in features_df.columns if col not in ['Nom', 'Cote', 'Numéro de corde', 'Poids', 'Musique', 'Âge/Sexe', 'Jockey', 'Entraîneur']]
        
        return features_df[feature_columns], feature_columns
    
    def _parse_musique_score(self, musique):
        """Convertit la musique en score numérique"""
        if pd.isna(musique):
            return 0.5
        
        try:
            # Exemple: "1a2a3a" -> moyenne des positions
            positions = []
            for char in str(musique):
                if char.isdigit():
                    positions.append(int(char))
            
            if positions:
                avg_position = np.mean(positions)
                return 1 / avg_position  # Meilleure position = score plus élevé
            return 0.5
        except:
            return 0.5

    def train_model(self, features, labels):
        """Entraîne le modèle de machine learning sélectionné"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=ML_CONFIG["test_size"], 
            random_state=ML_CONFIG["random_state"],
            stratify=labels
        )
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_type = ML_CONFIG["model_type"]
        
        if model_type == "logistic":
            self.model = LogisticRegression(
                random_state=ML_CONFIG["random_state"],
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=ML_CONFIG["random_state"],
                class_weight='balanced',
                max_depth=6
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=ML_CONFIG["random_state"],
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if sum(y_train) > 0 else 1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=ML_CONFIG["random_state"],
                max_depth=5
            )
        
        # Entraînement du modèle
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation
        train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
        test_probs = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.performance_metrics = {
            'train_auc': roc_auc_score(y_train, train_probs),
            'test_auc': roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else 0.5,
            'train_log_loss': log_loss(y_train, train_probs),
            'test_log_loss': log_loss(y_test, test_probs) if len(np.unique(y_test)) > 1 else float('inf'),
            'feature_importance': self._get_feature_importance(features.columns)
        }
        
        return self.performance_metrics

    def _get_feature_importance(self, feature_names):
        """Extrait l'importance des features"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
            
        feature_importance = dict(zip(feature_names, importances))
        return {k: v for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}

    def predict_proba(self, features):
        """Prédit les probabilités pour de nouvelles données"""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        
        # Calibration simple
        if ML_CONFIG["calibration"]:
            probabilities = self._calibrate_probabilities(probabilities)
            
        return probabilities

    def _calibrate_probabilities(self, probabilities):
        """Calibre les probabilités pour qu'elles soient plus réalistes"""
        # Évite les probabilités extrêmes
        probabilities = np.clip(probabilities, 0.01, 0.99)
        
        # Normalisation pour que la somme soit ~1 dans une course
        if len(probabilities) > 1:
            probabilities = probabilities / probabilities.sum() * min(len(probabilities) * 0.15, 0.95)
            
        return probabilities

# ==== FONCTIONS EXISTANTES AMÉLIORÉES ====
def prepare_features_ml(df, predictor):
    """Version améliorée de prepare_features avec engineering de features ML"""
    print("\n🔧 PRÉPARATION DES DONNÉES POUR ML...")
    
    # Conversion sécurisée des cotes
    df['odds_numeric'] = df['Cote'].apply(safe_float_convert)
    
    # Conversion des numéros de corde
    df['draw_numeric'] = df['Numéro de corde'].apply(safe_int_convert)
    
    # Extraction du poids en kg
    df['weight_kg'] = df['Poids'].apply(extract_weight_kg)
    
    # Nettoyage des données
    initial_count = len(df)
    df = df.dropna(subset=['odds_numeric', 'draw_numeric'])
    df['weight_kg'] = df['weight_kg'].fillna(df['weight_kg'].mean())
    final_count = len(df)
    
    print(f"✅ Données nettoyées : {initial_count} → {final_count} chevaux")
    
    # Engineering de features ML
    features_df, feature_names = predictor.engineer_features(df)
    
    return df, features_df, feature_names

def analyze_race_ml(df, race_type="AUTO", use_ml=True):
    """Analyse avec modèle ML intégré"""
    
    # Initialisation du prédicteur
    predictor = HorseRacingPredictor()
    
    # Préparation des données
    df_clean, features_df, feature_names = prepare_features_ml(df, predictor)
    
    if use_ml and len(df_clean) >= 6:  # Minimum 6 chevaux pour ML
        print("\n🤖 ENTRAÎNEMENT DU MODÈLE ML...")
        
        # Création de labels synthétiques
        labels = predictor.create_synthetic_labels(df_clean)
        
        if sum(labels) > 1:  # Au moins 2 positifs pour l'entraînement
            # Entraînement du modèle
            metrics = predictor.train_model(features_df, labels)
            
            # Prédictions
            ml_probabilities = predictor.predict_proba(features_df)
            df_clean['ml_probability'] = ml_probabilities
            df_clean['ml_score'] = ml_probabilities * 100
            
            print(f"✅ Modèle ML entraîné - AUC: {metrics['test_auc']:.3f}")
            
            # Affichage des features importantes
            top_features = list(metrics['feature_importance'].keys())[:5]
            print(f"📊 Top features: {', '.join(top_features)}")
            
        else:
            print("⚠️ Données insuffisantes pour l'entraînement ML, utilisation méthode classique")
            use_ml = False
    
    if not use_ml or len(df_clean) < 6:
        # Fallback vers méthode classique
        print("🔄 Utilisation de la méthode classique...")
        df_clean = analyze_race_classic(df_clean, race_type)
        df_clean['ml_probability'] = df_clean['score_final_normalized']
        df_clean['ml_score'] = df_clean['score_final'] * 10
    
    # Classement final
    df_ranked = df_clean.sort_values('ml_probability', ascending=False).reset_index(drop=True)
    df_ranked['rang_ml'] = range(1, len(df_ranked) + 1)
    
    return df_ranked, predictor if use_ml else None

def analyze_race_classic(df, race_type):
    """Méthode classique comme fallback"""
    # Implémentation simplifiée de votre méthode actuelle
    df['score_odds'] = 1 / df['odds_numeric']
    df['score_draw'] = 1 / df['draw_numeric']
    df['score_weight'] = 1 / df['weight_kg']
    
    df['score_final'] = (
        0.6 * df['score_odds'] + 
        0.25 * df['score_draw'] + 
        0.15 * df['score_weight']
    )
    
    # Normalisation
    df['score_final_normalized'] = (
        df['score_final'] - df['score_final'].min()
    ) / (df['score_final'].max() - df['score_final'].min())
    
    return df

def generate_ml_report(df_ranked, predictor, race_type):
    """Génère un rapport détaillé avec insights ML"""
    
    report = []
    report.append("🤖 RAPPORT D'ANALYSE AVEC MACHINE LEARNING")
    report.append("=" * 60)
    
    # Métriques de performance si disponible
    if predictor and hasattr(predictor, 'performance_metrics'):
        metrics = predictor.performance_metrics
        report.append(f"📈 PERFORMANCE DU MODÈLE ({ML_CONFIG['model_type'].upper()})")
        report.append(f"   • AUC (test): {metrics['test_auc']:.3f}")
        report.append(f"   • Log Loss (test): {metrics['test_log_loss']:.3f}")
        
        # Features importantes
        if metrics['feature_importance']:
            report.append(f"   • Feature principale: {list(metrics['feature_importance'].keys())[0]}")
    
    report.append(f"\n🎯 STRATÉGIE {race_type.replace('_', ' ')}:")
    
    if race_type == "PLAT":
        report.append("   • Modèle ML pondère: cotes, poids, corde + interactions")
        report.append("   • Features clés: déviation poids, position corde, log(cotes)")
    elif "ATTELE" in race_type:
        report.append("   • Focus sur: cotes réciproques, patterns performance")
        report.append("   • Interactions cotes/position importantes")
    
    # Analyse du top 3 ML
    report.append(f"\n🔍 ANALYSE TOP 3 ML:")
    
    for i in range(min(3, len(df_ranked))):
        cheval = df_ranked.iloc[i]
        prob_percent = cheval['ml_probability'] * 100
        
        analysis = []
        if cheval['odds_numeric'] < 5.0:
            analysis.append("cote très faible")
        elif cheval['odds_numeric'] < 10.0:
            analysis.append("cote intéressante")
            
        if cheval.get('draw_numeric', 10) <= 4:
            analysis.append("bonne position")
            
        if prob_percent > 20:
            analysis.append("haute confiance ML")
        elif prob_percent > 10:
            analysis.append("confiance modérée")
            
        report.append(f"   {i+1}. {cheval['Nom']} → {prob_percent:.1f}% ({', '.join(analysis)})")
    
    # Recommandations paris
    report.append(f"\n💡 RECOMMANDATIONS PARIS:")
    
    # Identification des valeurs
    value_picks = df_ranked[
        (df_ranked['ml_probability'] > df_ranked['ml_probability'].quantile(0.7)) &
        (df_ranked['odds_numeric'] > df_ranked['odds_numeric'].median())
    ].head(3)
    
    if len(value_picks) > 0:
        report.append("   💎 VALEURS DÉTECTÉES (bonne proba + cote élevée):")
        for _, pick in value_picks.iterrows():
            report.append(f"      • {pick['Nom']} (Proba: {pick['ml_probability']*100:.1f}%, Cote: {pick['odds_numeric']:.1f})")
    else:
        report.append("   🎯 FAVORIS LOGIQUES (haute probabilité ML):")
        for i in range(min(3, len(df_ranked))):
            cheval = df_ranked.iloc[i]
            report.append(f"      • {cheval['Nom']} (Proba: {cheval['ml_probability']*100:.1f}%)")
    
    return "\n".join(report)

# ==== STREAMLIT APP ====
def main():
    st.set_page_config(
        page_title="Pronostics Hippiques ML",
        page_icon="🏇",
        layout="wide"
    )
    
    st.title("🏇 Pronostics Hippiques avec Machine Learning")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration ML")
    
    ml_model = st.sidebar.selectbox(
        "Modèle de prédiction",
        ["xgboost", "logistic", "random_forest", "gradient_boosting"],
        index=0
    )
    
    target_var = st.sidebar.selectbox(
        "Variable cible",
        ["top3", "winner"],
        index=0
    )
    
    ML_CONFIG.update({
        "model_type": ml_model,
        "target_variable": target_var
    })
    
    # Section URL input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url = st.text_input("🔗 URL de la course (Geny.fr):", placeholder="https://www.geny.com/...")
    
    with col2:
        race_type = st.selectbox(
            "Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"],
            index=0
        )
    
    if st.button("🎯 Analyser la course", type="primary"):
        if url:
            with st.spinner("Analyse en cours avec ML..."):
                try:
                    # Récupération des données (votre code existant)
                    response = requests.get(url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    donnees_chevaux = []
                    
                    # Extraction des données (adaptez selon votre logique existante)
                    table = soup.find('table')
                    if table:
                        rows = table.find_all('tr')[1:]
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) >= 8:
                                donnees_chevaux.append({
                                    "Numéro de corde": nettoyer_donnees(cols[0].text),
                                    "Nom": nettoyer_donnees(cols[1].text),
                                    "Musique": nettoyer_donnees(cols[5].text) if len(cols) > 5 else "",
                                    "Âge/Sexe": nettoyer_donnees(cols[6].text) if len(cols) > 6 else "",
                                    "Poids": nettoyer_donnees(cols[7].text) if len(cols) > 7 else "60.0",
                                    "Jockey": nettoyer_donnees(cols[8].text) if len(cols) > 8 else "",
                                    "Entraîneur": nettoyer_donnees(cols[9].text) if len(cols) > 9 else "",
                                    "Cote": nettoyer_donnees(cols[-1].text)
                                })
                    
                    if donnees_chevaux:
                        df = pd.DataFrame(donnees_chevaux)
                        
                        # Analyse avec ML
                        df_ranked, predictor = analyze_race_ml(df, race_type)
                        
                        # Affichage des résultats
                        st.success(f"✅ Analyse terminée - {len(df_ranked)} chevaux analysés")
                        
                        # Tableau des résultats
                        st.subheader("📊 Classement ML")
                        
                        display_cols = ['rang_ml', 'Nom', 'ml_probability', 'Cote', 'Numéro de corde', 'Poids']
                        display_df = df_ranked[display_cols].copy()
                        display_df['Probabilité'] = (display_df['ml_probability'] * 100).round(1).astype(str) + '%'
                        display_df = display_df.rename(columns={
                            'rang_ml': 'Rang',
                            'Nom': 'Cheval', 
                            'Cote': 'Cote',
                            'Numéro de corde': 'Corde',
                            'Poids': 'Poids'
                        })
                        
                        st.dataframe(
                            display_df[['Rang', 'Cheval', 'Probabilité', 'Cote', 'Corde', 'Poids']],
                            use_container_width=True
                        )
                        
                        # Rapport détaillé
                        st.subheader("📈 Analyse détaillée")
                        report = generate_ml_report(df_ranked, predictor, race_type)
                        st.text_area("Rapport d'analyse", report, height=300)
                        
                        # Recommendations paris
                        st.subheader("🎯 Recommendations pour Paris")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("🥇 Premier favori", 
                                     f"{df_ranked.iloc[0]['Nom']}", 
                                     f"{df_ranked.iloc[0]['ml_probability']*100:.1f}%")
                        
                        with col2:
                            st.metric("💎 Meilleure valeur", 
                                     "À déterminer", 
                                     "Analyse avancée")
                        
                        with col3:
                            st.metric("📈 Confiance modèle", 
                                     f"{predictor.performance_metrics['test_auc']:.3f}" if predictor else "N/A", 
                                     "AUC Score")
                    
                    else:
                        st.error("❌ Aucune donnée extraite de l'URL fournie")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        else:
            st.warning("⚠️ Veuillez entrer une URL valide")

    # Section démo avec données exemple
    with st.expander("🧪 Tester avec des données exemple"):
        if st.button("Charger des données de démo"):
            # Données de démo
            demo_data = {
                'Nom': ['Star Runner', 'Thunder Bolt', 'Wind Dancer', 'Lightning Flash', 'Storm Chaser'],
                'Numéro de corde': ['3', '8', '1', '12', '5'],
                'Cote': ['4.5', '8.2', '3.8', '15.0', '6.5'],
                'Poids': ['56.0', '58.5', '55.0', '60.0', '57.0'],
                'Musique': ['1a2a', '3a1a', '2a2a', '5a4a', '2a3a']
            }
            
            df_demo = pd.DataFrame(demo_data)
            df_ranked, predictor = analyze_race_ml(df_demo, "PLAT")
            
            st.dataframe(df_ranked[['rang_ml', 'Nom', 'ml_probability', 'Cote']].head())

# ==== FONCTIONS UTILITAIRES EXISTANTES ====
def safe_float_convert(value):
    """Conversion sécurisée vers float"""
    if pd.isna(value):
        return np.nan
    try:
        cleaned = str(value).replace(',', '.').strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return np.nan

def safe_int_convert(value):
    """Conversion sécurisée vers entier"""
    if pd.isna(value):
        return np.nan
    try:
        cleaned = re.search(r'\d+', str(value))
        return int(cleaned.group()) if cleaned else np.nan
    except (ValueError, AttributeError):
        return np.nan

def extract_weight_kg(poids_str):
    """Extrait le poids en kg depuis une chaîne"""
    if pd.isna(poids_str):
        return np.nan
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    if match:
        return float(match.group(1).replace(',', '.'))
    return np.nan

def nettoyer_donnees(ligne):
    """Fonction de nettoyage héritée"""
    ligne = ''.join(e for e in ligne if e.isalnum() or e.isspace() or e in ['.', ',', '-', '(', ')', '%'])
    return ligne.strip()

if __name__ == "__main__":
    main()
