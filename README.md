# Projet de Graphe Neural Network

**Sujet:** Etat de l’art des réseaux de convolutions de graphes (via Pytorch geometrics) et leurs nuances.

**Auteurs:** Cléa HAN - Léa TRIQUET - Adrien ZABBAN

Ce projet se penche sur trois des méthodes les plus prometteuses et influentes dans le domaine du deep learning avec des graphes :
- Graph Convolusional Networks (GCN),
- Simplifying Graph Convolutional Networks (SGC) 
- Graph Isomorphism Network (GIN)
  

# Contenu
Ce projet contient les dossiers et fichiers suivants :

- `/data` : dossier contenant les données ENZYMES et MUTAG
- `/report` : dossier contenant le rapport en LaTeX (version PDF et TEX)
- `/results` : dossier contenant les résultats des expériences. Chaque sous-dossier contient :
    - les informations de l'expérience (`config.yaml`)
    - les courbes d'apprentissage (`loss.png` et `acc.png`)
    - les poids des modèles (`checkpoint.pt`)
    - l'ensemble des valeurs de loss et métriques (`train_log.csv`)
    - résultats de test (`test.txt`)
- `dataloader.py` qui permet de récupérer les graphes
- `model.py` qui permet de récupérer le réseau de neurones voulu
- `train.py` qui entraîne un modèle et sauvegarde le résultat dans `/results`
- `test.py` qui teste une expérience et sauvegarde le résultat dans `/results`
- `utils.py` qui s'occupe de gérer les sauvegardes de l'entraînement et du test
- `requiements.txt` qui liste l'ensemble des packages python à avoir avec leur version pour le bon fonctionement du code (on a utlisé python 3.9). Vous pouvez utiliser la commande `pip install -r requiements.txt` pour tout installer

