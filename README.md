# BVMT Stocks Processed Data

Ce dépôt contient des données boursières traitées pour les actions de la BVMT (Bourse des Valeurs Mobilières de Tunis).

## Structure du fichier CSV

Le fichier `data/bvmt_stocks_processed.csv` contient les colonnes suivantes :

1. **Date** : Date de l'enregistrement (AAAA-MM-JJ)
2. **Company** : Symbole boursier de l'entreprise
3. **Sector** : Secteur d'activité
4. **Open** : Prix d'ouverture
5. **Close** : Prix de clôture
6. **High** : Prix le plus haut de la journée
7. **Low** : Prix le plus bas de la journée
8. **Volume** : Volume d'échanges
9. **Change %** : Variation en pourcentage du prix de clôture par rapport à la veille
10. **MA (Moving Average)** : Moyenne mobile des prix
11. **Volatility** : Volatilité (écart-type)
12. **Avg Volume** : Volume moyen d'échanges
13. **momentum_3d** :Mesure le taux de variation des prix sur trois périodes, indiquant la force d'une tendance.
14. **price_accel** :  Suit le taux de changement des prix, mettant en évidence l'accélération ou la décélération du mouvement des prix
15. **rel_sector_strength** :Compare la performance d'un secteur à celle du marché global, indiquant sa force relative.






## Utilisation

Vous pouvez utiliser ce jeu de données pour l'analyse boursière, la modélisation prédictive, ou l'apprentissage automatique.

## Structure du projet

```
BVMT-Scraping/
├── data/
│   └── bvmt_stocks_processed.csv      # Ton fichier de données principal
├── scripts/
│   └── example_analysis.py            # (Optionnel) Exemple de script d'analyse ou de chargement des données
├── README.md                          # Présentation du projet, explications, structure des données
├── requirements.txt                   # Dépendances Python nécessaires
├── .gitignore                         # Fichiers à ignorer par git
└── LICENSE                            # (Optionnel mais recommandé) Licence d'utilisation
```

## Licence

Ce projet est fourni à des fins éducatives et de recherche. 
