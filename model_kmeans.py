import pickle

class KMeansModel:
    def __init__(self, model_path):
        """Initializes the KMeansModel with the provided model_path."""
        with open(model_path, 'rb') as model_file:
            self.kmeans_proteine = pickle.load(model_file)
            self.kmeans_amidon = pickle.load(model_file)
            self.kmeans_calcium = pickle.load(model_file)

    def predict(self, proteine, amidon, calcium):
        """Predicts the cluster for the given input."""
        cluster_proteine = int(self.kmeans_proteine.predict([[proteine]])[0])
        cluster_amidon = int(self.kmeans_amidon.predict([[amidon]])[0])
        cluster_calcium = int(self.kmeans_calcium.predict([[calcium]])[0])
        
        # Retourner un dictionnaire avec des valeurs converties en int
        prediction = {
            "Cluster_Proteine": cluster_proteine,
            "Cluster_Amidon": cluster_amidon,
            "Cluster_Calcium": cluster_calcium
        }
        return prediction
