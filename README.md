# explainerdashboard via heroku
by: Enzo Etelbert

Repo permettant la mise en ligne d'un explainerdashboard

 ## Installation

TODO

## 1ere étape: apprentissage

Adapter et lancer training.py

## 2eme étape: Construction de l'explainer

Lancer main.py afin de générer l'explainer. A réadapter également.

## 3eme étape: Heroku

Créer un login si ce n'est pas déjà fait sur https://id.heroku.com/login 

Ensuite créer une nouvelle application en cliquant sur new -> create a new app
Donner un nom (ici noté $name-app$) et sélectionner Europe
Ouvrir un terminal et se placer sur le répertoire cbe_app
Taper "heroku login" et cliquer sur Log In sur la page web qui s'est ouverte
Taper la commande "heroku git:remote -a $name-app$"
Enfin taper la commande "git push heroku master"