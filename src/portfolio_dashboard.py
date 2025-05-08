#!/usr/bin/env python
"""
Pipeline d'Analyse du Marché Boursier BVMT - Tableau de Bord Portfolio
"""
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import os
import base64
import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import sys
import glob

# Ajouter le répertoire racine du projet au chemin
sys.path.append('.')

# Initialiser l'application Dash
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}]
               )

app.title = "Portfolio d'Analyse du Marché Boursier BVMT"

# Fonction pour lire les fichiers texte
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Erreur de lecture du fichier: {str(e)}"

# Fonction pour charger et prévisualiser les données CSV
def load_csv_preview(file_path, rows=5):
    try:
        df = pd.read_csv(file_path)
        return df.head(rows)
    except Exception as e:
        return pd.DataFrame({'Erreur': [str(e)]})

# Lire les fichiers d'analyse
market_summary = read_text_file('data/insights/market_summary.txt')
company_analysis = read_text_file('data/insights/company_analysis.txt')
sector_classification = read_text_file('data/insights/sector_classification.txt')
data_insights = read_text_file('data/insights/data_insights.txt')

# Obtenir les fichiers de visualisation disponibles
viz_files = glob.glob('data/plots/*.html')
viz_file_names = [os.path.basename(f) for f in viz_files]

# Charger les prévisualisations CSV
raw_data_preview = None
processed_data_preview = None

if os.path.exists('data/bvmt_stocks.csv'):
    raw_data_preview = load_csv_preview('data/bvmt_stocks.csv')
if os.path.exists('data/bvmt_stocks_processed.csv'):
    processed_data_preview = load_csv_preview('data/bvmt_stocks_processed.csv')

# Structure du projet
project_structure = """
Pipeline_Analyse_Marché_Boursier_BVMT/
│
├── data/
│   ├── bvmt_stocks.csv                   # Données brutes des actions (scrapées ou exemple)
│   ├── bvmt_stocks_processed.csv         # Données traitées pour l'analyse
│   ├── insights/                         # Répertoire pour les analyses générées
│   │   ├── market_summary.txt            # Résumé du marché
│   │   ├── company_analysis.txt          # Analyse spécifique des entreprises
│   │   ├── sector_classification.txt     # Classification des secteurs
│   │   └── data_insights.txt             # Aperçus généraux des données
│   ├── plots/                            # Répertoire pour les visualisations générées
│   │   ├── sector_heatmap.html           # Carte thermique des performances sectorielles
│   │   ├── sector_volume.html            # Visualisation du volume d'échanges par secteur
│   │   ├── volume_vs_price.html          # Visualisation volume vs prix
│   │   └── <entreprise>_price_volume.html # Graphique prix et volume spécifique à une entreprise
│   └── models/                           # Répertoire pour les modèles sauvegardés
│       ├── <entreprise>_linear_model.pkl # Modèle de régression linéaire pour une entreprise
│       ├── <entreprise>_rf_model.pkl     # Modèle de forêt aléatoire pour une entreprise
│       └── <entreprise>_predictions.html # Visualisation des prédictions du modèle
│
├── src/
│   ├── app.py                            # Fichier principal de l'application pour le tableau de bord
│   ├── scraper.py                        # Module pour le scraping des données boursières
│   ├── preprocessing.py                  # Module pour le prétraitement des données
│   ├── visualization.py                  # Module pour générer des visualisations
│   ├── model.py                          # Module pour l'entraînement et les prédictions avec des modèles ML
│   └── llm_local.py                      # Module pour l'analyse LLM locale
│
├── requirements.txt                      # Liste des dépendances Python
├── README.md                             # Aperçu du projet et instructions
└── main.py                               # Point d'entrée pour exécuter l'ensemble du pipeline
"""

# Stack technologique
tech_stack = {
    "Langages": ["Python"],
    "Manipulation de données": ["Pandas", "NumPy"],
    "Visualisation": ["Plotly", "Dash", "Dash Bootstrap Components"],
    "Apprentissage automatique": ["Scikit-learn", "TensorFlow/PyTorch"],
    "LLM": ["Transformers (HuggingFace)", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
    "Technologies Web": ["HTML", "CSS", "Flask (sous-jacent à Dash)"]
}

# Créer la mise en page de l'application
app.layout = dbc.Container([
    # En-tête avec le titre du projet
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("Pipeline d'Analyse du Marché Boursier BVMT", className="display-4 text-center mt-4"),
                html.H4("Un Outil Complet pour l'Analyse du Marché Boursier Tunisien", className="text-center mb-4 text-muted"),
                html.Hr()
            ])
        ])
    ]),
    
    # Aperçu du projet et objectifs
    dbc.Row([
        dbc.Col([
            html.H2("Aperçu du Projet", className="mt-4 mb-3"),
            html.P([
                "Le Pipeline d'Analyse du Marché Boursier BVMT est un outil complet conçu pour analyser et visualiser les données ",
                "du marché boursier de la Bourse des Valeurs Mobilières de Tunis (BVMT). Ce projet intègre divers composants, y compris ",
                "le scraping de données, le prétraitement, la visualisation, la modélisation par apprentissage automatique, et l'analyse ",
                "par modèle de langage local (LLM), pour fournir aux utilisateurs des insights exploitables sur les tendances du marché ",
                "et la performance individuelle des actions."
            ]),
            html.H4("Objectifs du Projet", className="mt-4 mb-3"),
            html.Ul([
                html.Li("Automatiser la collecte des données du marché boursier"),
                html.Li("Prétraiter et nettoyer les données pour l'analyse"),
                html.Li("Visualiser la performance des actions et les tendances du marché de manière interactive"),
                html.Li("Implémenter des modèles d'apprentissage automatique pour la prédiction des prix"),
                html.Li("Générer des insights en langage naturel à l'aide d'un LLM local")
            ]),
        ], width=12)
    ]),
    
    # Structure du projet
    dbc.Row([
        dbc.Col([
            html.H2("Structure du Projet", className="mt-4 mb-3"),
            html.P("Le projet est organisé selon la structure de répertoire suivante:"),
            html.Div([
                dcc.Markdown(f"```\n{project_structure}\n```")
            ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'fontFamily': 'monospace'})
        ], width=12)
    ]),
    
    # Étapes du Pipeline
    dbc.Row([
        dbc.Col([
            html.H2("Étapes du Pipeline", className="mt-4 mb-3"),
            dbc.Tabs([
                dbc.Tab([
                    html.H4("Scraping de Données", className="mt-3"),
                    html.P([
                        "La première étape du pipeline consiste à collecter des données du marché boursier. ",
                        "Le module de scraping se connecte au site Web de la BVMT pour recueillir des données de prix actuelles et historiques."
                    ]),
                    html.P([
                        "Si les données en temps réel ne sont pas disponibles, le système peut générer des données d'exemple à des fins de test."
                    ]),
                    html.H5("Aperçu des Données", className="mt-3"),
                    html.Div([
                        html.P("Données brutes (5 premières lignes):"),
                        dbc.Table.from_dataframe(raw_data_preview, striped=True, bordered=True, hover=True, size='sm') if raw_data_preview is not None else html.P("Aucune donnée brute disponible")
                    ], style={'overflowX': 'auto'})
                ], label="Scraping de Données"),
                
                dbc.Tab([
                    html.H4("Prétraitement des Données", className="mt-3"),
                    html.P([
                        "Le module de prétraitement nettoie et transforme les données brutes pour l'analyse. ",
                        "Cela inclut la gestion des valeurs manquantes, le calcul des indicateurs techniques et la normalisation des données."
                    ]),
                    html.H5("Aperçu des Données Traitées", className="mt-3"),
                    html.Div([
                        html.P("Données traitées (5 premières lignes):"),
                        dbc.Table.from_dataframe(processed_data_preview, striped=True, bordered=True, hover=True, size='sm') if processed_data_preview is not None else html.P("Aucune donnée traitée disponible")
                    ], style={'overflowX': 'auto'})
                ], label="Prétraitement"),
                
                dbc.Tab([
                    html.H4("Visualisation", className="mt-3"),
                    html.P([
                        "Le module de visualisation génère des graphiques et des diagrammes interactifs pour aider à comprendre les tendances du marché. ",
                        "En utilisant Plotly et Dash, les utilisateurs peuvent explorer les mouvements de prix, les volumes d'échange et la performance des secteurs."
                    ]),
                    html.H5("Visualisations Disponibles", className="mt-3"),
                    html.Ul([html.Li(file) for file in viz_file_names]) if viz_file_names else html.P("Aucun fichier de visualisation trouvé"),
                    html.P("Les visualisations peuvent être consultées dans le tableau de bord interactif.")
                ], label="Visualisation"),
                
                dbc.Tab([
                    html.H4("Modèles d'Apprentissage Automatique", className="mt-3"),
                    html.P([
                        "Le module de modélisation implémente des algorithmes d'apprentissage automatique pour prédire les prix futurs des actions. ",
                        "Il comprend des modèles de régression linéaire et de forêt aléatoire qui sont entraînés sur des données historiques."
                    ]),
                    html.H5("Fonctionnalités des Modèles", className="mt-3"),
                    html.Ul([
                        html.Li("Prédiction de prix utilisant des indicateurs techniques"),
                        html.Li("Analyse de l'importance des caractéristiques"),
                        html.Li("Évaluation des performances avec des métriques comme RMSE et MAE"),
                        html.Li("Prévision du prix du jour suivant")
                    ]),
                ], label="Modèles ML"),
                
                dbc.Tab([
                    html.H4("Analyse LLM Locale", className="mt-3"),
                    html.P([
                        "Le module LLM utilise des modèles de langage locaux pour générer des insights et des analyses en langage naturel. ",
                        "Contrairement aux solutions qui dépendent d'API externes, cette approche garantit la confidentialité et élimine les coûts d'API."
                    ]),
                    html.H5("Fonctionnalités LLM", className="mt-3"),
                    html.Ul([
                        html.Li("Résumés de marché basés sur des données récentes"),
                        html.Li("Explications des mouvements de prix spécifiques aux entreprises"),
                        html.Li("Classification et analyse des secteurs"),
                        html.Li("Insights généraux sur les tendances du marché")
                    ]),
                    html.P("Le module LLM utilise des modèles comme TinyLlama pour l'inférence locale.")
                ], label="Analyse LLM"),
            ]),
        ], width=12)
    ]),
    
    # Exemples d'Insights
    dbc.Row([
        dbc.Col([
            html.H2("Insights Générés par LLM", className="mt-4 mb-3"),
            dbc.Tabs([
                dbc.Tab([
                    html.H4("Résumé du Marché", className="mt-3"),
                    html.P("Exemple de sortie de l'analyse du marché par LLM local:"),
                    html.Div([
                        dcc.Markdown(market_summary.replace('#', '##'))
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'maxHeight': '400px', 'overflowY': 'auto'})
                ], label="Résumé du Marché"),
                
                dbc.Tab([
                    html.H4("Analyse d'Entreprise", className="mt-3"),
                    html.P("Exemple de sortie de l'analyse d'entreprise par LLM local:"),
                    html.Div([
                        dcc.Markdown(company_analysis.replace('#', '##'))
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'maxHeight': '400px', 'overflowY': 'auto'})
                ], label="Analyse d'Entreprise"),
                
                dbc.Tab([
                    html.H4("Classification des Secteurs", className="mt-3"),
                    html.P("Exemple de sortie de la classification des secteurs par LLM local:"),
                    html.Div([
                        dcc.Markdown(sector_classification.replace('#', '##'))
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'maxHeight': '400px', 'overflowY': 'auto'})
                ], label="Classification des Secteurs"),
                
                dbc.Tab([
                    html.H4("Insights sur les Données", className="mt-3"),
                    html.P("Exemple de sortie des insights sur les données par LLM local:"),
                    html.Div([
                        dcc.Markdown(data_insights.replace('#', '##'))
                    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'maxHeight': '400px', 'overflowY': 'auto'})
                ], label="Insights sur les Données"),
            ]),
        ], width=12)
    ]),
    
    # Aperçu du Tableau de Bord
    dbc.Row([
        dbc.Col([
            html.H2("Tableau de Bord Interactif", className="mt-4 mb-3"),
            html.P([
                "Le projet comprend un tableau de bord interactif qui permet aux utilisateurs d'explorer les données et les visualisations. ",
                "Voici comment y accéder:"
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Lancer le Tableau de Bord", className="card-title"),
                    html.P("Exécutez la commande suivante depuis la racine du projet:"),
                    html.Pre("python main.py --dashboard", className="bg-light p-2"),
                    html.P("Puis ouvrez votre navigateur et accédez à:"),
                    html.Pre("http://127.0.0.1:8050/", className="bg-light p-2"),
                ])
            ], className="mb-4"),
            html.P([
                "Le tableau de bord offre des options pour sélectionner des entreprises, des plages de dates et différents types d'analyse. ",
                "Les utilisateurs peuvent générer des insights sur le marché et visualiser les données de manière interactive."
            ]),
        ], width=12)
    ]),
    
    # Stack Technologique
    dbc.Row([
        dbc.Col([
            html.H2("Stack Technologique", className="mt-4 mb-3"),
            html.Div([
                html.Div([
                    html.H5(category, className="mt-3"),
                    html.Ul([html.Li(tech) for tech in tech_list])
                ]) for category, tech_list in tech_stack.items()
            ])
        ], width=12)
    ]),
    
    # Conclusion et Améliorations Futures
    dbc.Row([
        dbc.Col([
            html.H2("Conclusion et Améliorations Futures", className="mt-4 mb-3"),
            html.P([
                "Le Pipeline d'Analyse du Marché Boursier BVMT fournit un outil complet pour analyser le marché boursier tunisien. ",
                "Il combine le scraping de données, le prétraitement, la visualisation, l'apprentissage automatique et l'analyse LLM locale ",
                "pour offrir des insights précieux aux investisseurs et analystes."
            ]),
            html.H4("Améliorations Futures", className="mt-3"),
            html.Ul([
                html.Li("Implémenter des mises à jour de données en temps réel pour l'analyse du marché en direct"),
                html.Li("Explorer des techniques d'apprentissage automatique avancées pour améliorer les prédictions"),
                html.Li("Améliorer les capacités LLM avec des modèles financiers plus spécialisés"),
                html.Li("Étendre le tableau de bord pour inclure des métriques financières supplémentaires"),
                html.Li("Ajouter des fonctionnalités d'optimisation de portefeuille et d'analyse de risques")
            ]),
        ], width=12)
    ]),
    
    # Pied de page
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Portfolio du Pipeline d'Analyse du Marché Boursier BVMT - Créé à des fins éducatives", className="text-center text-muted mb-4")
        ])
    ]),
    
], fluid=True, className="mb-5")

if __name__ == '__main__':
    app.run(debug=True, port=8051) 