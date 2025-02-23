# KMeansClusterer

## Présentation
Ce projet implémente une classe `KMeansClusterer` en Python, permettant d'appliquer l'algorithme de clustering K-Means sur des matrices 3D extraites de fichiers `.mat`. L'objectif est d'organiser ces matrices en clusters afin d'analyser les schémas et dynamiques sous-jacents.

## Fonctionnalités
- Chargement et traitement des fichiers `.mat` contenant des matrices 3D.
- Conversion des matrices 3D en matrices 2D exploitables.
- Aplatissement et normalisation des matrices pour les rendre compatibles avec K-Means.
- Application de l'algorithme K-Means et stockage des résultats.
- Détermination du nombre optimal de clusters via :
  - La méthode du coude (Elbow Method).
  - Le coefficient de silhouette.
- Visualisation des centroïdes et des exemples de matrices par cluster.
- Affichage de la distribution des clusters.
- Attribution des clusters aux dyades et suivi de leur évolution temporelle.

## Prérequis
Avant d'exécuter le script, assurez-vous d'avoir installé les bibliothèques suivantes :

```bash
pip install numpy scipy matplotlib scikit-learn
```

## Guide d'utilisation

### 1. Initialisation de la classe
Créez une instance de la classe `KMeansClusterer` avec le nombre de clusters souhaité :

```python
from kmeans_clusterer import KMeansClusterer

n_clusters = 5  # Nombre de clusters souhaité
clusterer = KMeansClusterer(n_clusters)
```

### 2. Chargement des données
Chargez les fichiers `.mat` et générez les matrices exploitables :

```python
data_directory = "chemin/vers/le/repertoire"
data = clusterer.load_all_data(data_directory)
matrices = clusterer.generate_matrices(data)
```

### 3. Détermination du nombre optimal de clusters
Utilisez les méthodes de validation pour trouver le nombre optimal de clusters :

```python
clusterer.elbow_method(matrices, range_clusters=10)
clusterer.silhouette_score_method(matrices, range_clusters=10)
```

### 4. Exécution de l'algorithme K-Means
Appliquez K-Means sur les matrices et affichez le score de silhouette :

```python
clusterer.fit(matrices)
clusterer.print_silhouette_score()
```

### 5. Visualisation des résultats
Affichez les centroïdes, la distribution des clusters et l'évolution des dyades :

```python
clusterer.plot_centroids(matrices)
clusterer.cluster_distribution()
dyad_clusters = clusterer.assign_clusters_to_dyads()
clusterer.plot_dyad_clusters()
```

## Fichiers générés
Si un chemin de sauvegarde est spécifié, les résultats seront enregistrés sous forme d'images (`.png`) et de fichiers texte (`.txt`) dans le répertoire indiqué.


