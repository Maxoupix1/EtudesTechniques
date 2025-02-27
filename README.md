# KMeansClusterer

## Introduction
Ce projet propose une implémentation avancée de l'algorithme de clustering K-Means appliqué à des matrices 3D issues de fichiers `.mat`. Il permet une segmentation efficace des données afin d’extraire des motifs récurrents et d’étudier la dynamique des structures sous-jacentes. L’approche inclut des méthodes d’évaluation et de visualisation pour une interprétation approfondie des clusters générés.

## Fonctionnalités principales
- Chargement et traitement des fichiers `.mat` contenant des matrices 2D entre autres informations.
- Conversion des matrices contenues dans les fichiers `.mat` en représentations 2D exploitables pour le clustering.
- Prétraitement des données via aplatissement et normalisation.
- Application du clustering K-Means et stockage des résultats.
- Détermination du nombre optimal de clusters par :
  - La méthode du coude (Elbow Method), basée sur l’inertie intra-classe.
  - L’évaluation du coefficient de silhouette pour mesurer la cohésion et la séparation des clusters.
- Visualisation des centroïdes et échantillons représentatifs.
- Analyse de la distribution des clusters.
- Attribution des clusters aux dyades et étude de leur évolution temporelle.

## Prérequis techniques
L'exécution de ce projet nécessite l'installation des bibliothèques Python suivantes :

```bash
pip install numpy scipy matplotlib scikit-learn
```
Les versions des bibliothèques utilisées pour ce projet sont spécifiées dans `requirements.txt` en cas de besoin.

## Utilisation

### 1. Initialisation de la classe
Instanciez la classe `KMeansClusterer` en spécifiant le nombre de clusters souhaité :

```python
from kmeans_clusterer import KMeansClusterer

n_clusters = 5  # Définition du nombre de clusters
clusterer = KMeansClusterer(n_clusters)
```

### 2. Chargement et prétraitement des données
Importez les fichiers `.mat` et extrayez les matrices exploitables :

```python
data_directory = "chemin/vers/le/repertoire"
data = clusterer.load_all_data(data_directory)
matrices = clusterer.generate_matrices(data)
```

### 3. Sélection du nombre optimal de clusters
Utilisez les techniques d'évaluation pour choisir la valeur de k optimale :

```python
clusterer.elbow_method(matrices, range_clusters=10)
clusterer.silhouette_score_method(matrices, range_clusters=10)
```

### 4. Application de K-Means et analyse
Entraînez le modèle K-Means et obtenez le score de silhouette moyen :

```python
clusterer.fit(matrices)
clusterer.print_silhouette_score()
```

### 5. Visualisation et interprétation des résultats
Affichez les centroïdes, analysez la distribution des clusters et suivez leur évolution temporelle :

```python
clusterer.plot_centroids(matrices)
clusterer.cluster_distribution()
dyad_clusters = clusterer.assign_clusters_to_dyads()
clusterer.plot_dyad_clusters()
```

## Résultats et exportation
Les sorties générées peuvent être enregistrées sous forme d’images (`.png`) et de fichiers texte (`.txt`) pour une exploitation ultérieure. Si un chemin de sauvegarde est spécifié, les fichiers seront stockés dans le répertoire désigné.

## Dossier `notebooks pour traitement`
Ce dossier propose des exemples de modification et d'implémentation de la classe `KMeansClusterer` afin de n'utiliser que la diagonale ou d'utiiser un masque pour réaliser le clustering.

## Auteur et contexte
Ce projet a été développé dans un cadre académique pour fournir un outil robuste d’analyse de matrices 3D via le clustering K-Means. Il est destiné aux chercheurs et étudiants travaillant sur des problématiques de segmentation et d’extraction de motifs à partir de données multidimensionnelles.


