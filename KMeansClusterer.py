import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KMeansClusterer:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_labels = None
        self.centroids = None
        self.inertia = None
        self.matrices = None
        self.dyad_clusters = None

    def load_all_data(self, directory):
        """Charge toutes les matrices des fichiers .mat d'un répertoire"""
        all_data = []
        for file in os.listdir(directory):
            if file.endswith(".mat"):
                mat = scipy.io.loadmat(os.path.join(directory, file))
                mat_data = mat['WTC3DmatrixDyad']
                all_data.append(mat_data)
        return np.array(all_data)

    def generate_matrices(self, data):
        """Convertit les matrices 3D en une liste de matrices 2D exploitables"""
        matrices = [data[i, :, :, j] for i in range(data.shape[0]) for j in range(data.shape[3])]
        return np.array(matrices)

    def flatten_matrices(self, matrices):
        """Aplatie les matrices 2D en vecteurs"""
        return matrices.reshape(matrices.shape[0], -1)
    
    def extract_diagonals(self, matrices):
        """Extrait uniquement les diagonales des matrices et les transforme en vecteurs"""
        diagonals = [np.diagonal(matrix) for matrix in matrices]
        return np.array(diagonals)

    def fit(self, matrices):
        """Applique KMeans et stocke les résultats"""
        self.matrices = self.flatten_matrices(matrices)  # Sauvegarde des matrices originales aplaties
        self.matrices = self.matrices / self.matrices.max()  # Normalisation

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.cluster_labels = self.kmeans.fit_predict(self.matrices)
        self.centroids = self.kmeans.cluster_centers_

    def elbow_method(self, matrices, range_clusters, use_diagonal=False, save_path=None):
        """Trace la méthode du coude pour trouver le bon nombre de clusters"""
        if use_diagonal:
            data = self.extract_diagonals(matrices)
        else:
            data = self.flatten_matrices(matrices)


        flattened_data_normalized = data / data.max()

        inertia_values = []
        k_range = range(1, range_clusters)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(flattened_data_normalized)
            inertia_values.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertia_values, marker='o', linestyle='--')
        plt.xlabel('Nombre de clusters (k)')
        plt.ylabel('Inertie')
        plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
        plt.xticks(k_range)
        plt.grid()

        if save_path:
            plt.savefig(os.path.join(save_path, "elbow_method.png"))
        else:
            plt.show()
        plt.close()

    def silhouette_score_method(self, matrices, range_clusters, use_diagonal=False, save_path=None):
        """Trace le Silhouette Score en fonction du nombre de clusters"""
        if use_diagonal:
            data = self.extract_diagonals(matrices)
        else:
            data = self.flatten_matrices(matrices)
            
        flattened_data_normalized = data / data.max()

        silhouette_scores = []
        k_range = range(2, range_clusters)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(flattened_data_normalized)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(flattened_data_normalized, labels))

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
        plt.xlabel('Nombre de clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title("Silhouette Score pour déterminer le nombre optimal de clusters")
        plt.grid()
        
        if save_path:
            plt.savefig(os.path.join(save_path, "silhouette_method.png"))
        else:
            plt.show()
        plt.close()

    def plot_centroids(self, matrices, save_path=None):
        """Affiche les centroïdes des clusters ainsi que quelques exemples de matrices par cluster"""
        if self.kmeans is None:
            raise ValueError("Le modèle KMeans doit être entraîné avant d'afficher les centroïdes.")
        
        flattened_data = self.flatten_matrices(matrices)
        self.centroids = self.centroids.reshape(self.centroids.shape[0], matrices.shape[1], matrices.shape[2])

        plt.figure(figsize=(10, 5))
        for i in range(self.centroids.shape[0]):
            plt.subplot(2, self.centroids.shape[0], i + 1)
            plt.imshow(self.centroids[i], cmap='viridis', interpolation='nearest')
            plt.title(f'Centroïde {i+1}')
            plt.colorbar()
            plt.axis('off')

        # Visualisation de quelques exemples de matrices par cluster
        for i in range(self.centroids.shape[0]):
            cluster_indices = np.where(self.cluster_labels == i)[0]
            if len(cluster_indices) > 0:
                example_matrix = matrices[cluster_indices[0]]
                plt.subplot(2, self.centroids.shape[0], self.centroids.shape[0] + i + 1)
                plt.imshow(example_matrix, cmap='viridis', interpolation='nearest')
                plt.title(f'Exemple Cluster {i+1}')
                plt.colorbar()
                plt.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, "centroids.png"))
        else:
            plt.show()
        plt.close()

    def cluster_distribution(self, save_path=None):
        """Affiche et enregistre la distribution des éléments dans chaque cluster"""
        if self.cluster_labels is None:
            raise ValueError("Le modèle KMeans doit être entraîné avant d'afficher la distribution.")

        unique, counts = np.unique(self.cluster_labels, return_counts=True)

        # Construire le texte du rapport
        result_text = "Distribution des clusters :\n"
        for cluster, count in zip(unique, counts):
            result_text += f"Cluster {cluster}: {count} matrices\n"

        # Afficher dans la console
        print(result_text)

        # Si un chemin de sauvegarde est spécifié, enregistrer le fichier
        if save_path:
            file_path = os.path.join(save_path, "cluster_distribution.txt")
            with open(file_path, "w") as f:
                f.write(result_text)
            print(f"Distribution enregistrée dans {file_path}")
    
    def print_silhouette_score(self):
        """Calcule et affiche le Silhouette Score moyen après l'entraînement"""
        if self.kmeans is None or self.cluster_labels is None:
            raise ValueError("Le modèle KMeans doit être entraîné avant de calculer le Silhouette Score.")

        # Aplatir les matrices et normaliser les données
        flattened_data = self.flatten_matrices(self.matrices)  # Conserver les matrices originales
        flattened_data_normalized = flattened_data / flattened_data.max()

        # Calcul du score de silhouette
        score = silhouette_score(flattened_data_normalized, self.cluster_labels)
        print(f"Silhouette Score moyen: {score:.4f}")

    def assign_clusters_to_dyads(self):
        """Réassocie chaque ensemble de 16 matrices à sa dyade et lui assigne un cluster"""
        if self.cluster_labels is None:
            raise ValueError("Le modèle KMeans doit être entraîné avant d'assigner les clusters aux dyades.")

        dyad_clusters = {}
        for i, label in enumerate(self.cluster_labels):
            dyad = i // 16  # Supposition : 16 matrices par dyade
            if dyad not in dyad_clusters:
                dyad_clusters[dyad] = []
            dyad_clusters[dyad].append(label)

        self.dyad_clusters = dyad_clusters
        return dyad_clusters
    
    def plot_dyad_clusters(self, save_path=None):
        """Affiche l'évolution des clusters dans le temps pour chaque dyade"""
        if self.dyad_clusters is None:
            raise ValueError("Les dyades n'ont pas encore été associées aux clusters. Exécutez 'assign_clusters_to_dyads()' d'abord.")

        fig, axes = plt.subplots(len(self.dyad_clusters), 1, figsize=(10, 80), sharex=False)
        step_size = 120  # Durée entre chaque point

        time = np.arange(len(next(iter(self.dyad_clusters.values())))) * step_size  # Axe des X

        for i, (dyad, values) in enumerate(self.dyad_clusters.items()):
            unique_values = sorted(set(values))  # Détermine les valeurs uniques des clusters pour l'axe des Y
            axes[i].step(time, values, where='post', label=f"Dyade {dyad}")
            axes[i].set_ylim(min(unique_values) - 0.5, max(unique_values) + 0.5)
            axes[i].set_yticks(unique_values)
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend()

        plt.xlabel("Temps en (s)")
        
        # Si un chemin de sauvegarde est spécifié, enregistrer le fichier
        if save_path:
            plt.savefig(os.path.join(save_path, "dyad_cluster_over_time.png"))
        else:
            plt.show()
        plt.close()