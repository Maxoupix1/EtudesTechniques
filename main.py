from KMeansClusterer import KMeansClusterer
import os

# Définir un dossier de sauvegarde
save_directory = "Results\\results_60_5"
os.makedirs(save_directory, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Charger et préparer les données
clusterer = KMeansClusterer(n_clusters=5)
data = clusterer.load_all_data("Data\\Win60sec-Overlap5sec")
matrices = clusterer.generate_matrices(data)

# Générer et sauvegarder les plots
clusterer.elbow_method(matrices, range_clusters=10, save_path=save_directory)
clusterer.silhouette_score_method(matrices, range_clusters=10, save_path=save_directory)
clusterer.fit(matrices)
clusterer.plot_centroids(matrices, save_path=save_directory)
clusterer.print_silhouette_score()
clusterer.assign_clusters_to_dyads()
clusterer.plot_dyad_clusters(save_path=save_directory)
