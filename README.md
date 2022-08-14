# Projet7-Openclassrooms
Parcours Data Science
by: Enzo Etelbert

## Links : 
### App : https://enzism-p7-v2-dashboard-front-c14yis.streamlitapp.com
### Git : https://github.com/enzism/P7_v2

Projet n°7 : "Implémentez un modèle de scoring"

## Description du projet
* Supervised learning sur un jeu de données déséquilibré (pénalisation des classes par Sample Weights et SMOTE)
* Choix d'une métrique adaptée à un problème métier (F Beta Score)
* Construction d'un modèle de scoring supervisé
* Mise en place d'une API Flask déployée sur heroku pour appeler le modèle de prédiction 
* Construction d'un dashboard interactif à destination des gestionnaires de relation client (Streamlit)
* Utilisation d'un logiciel de versionning : github ou gitflow

# Streamlit dashboard and API via Streamlit share and heroku respectively


Repo permettant la mise en ligne d'un dashboard interactif

 ## Installation

pip install -r requirements.txt

## 1ere étape: apprentissage

Adapter et lancer preprocessing.py puis training.py qui sont les codes de préparant les données et entraînant le modèle LGBMClassifier de Microsoft. 

## 2eme étape: Déploiement de l'API de prédiction

Lancer app.py avec l'url locale ou le déployer sur heroku afin d'y accéder depuis n'importe quel poste en publique.

## 3eme étape: Streamlit

Dashboard intéractif déployé sur streamlit et possédant une connexion via une API heroku (app.py) pour réaliser la prédiction du modèle.


## Autres livrables :

Deux notebooks permettant de comprendre la démarche de preprocessing et du choix du modèle.