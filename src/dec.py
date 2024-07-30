from sklearn.cluster import KMeans
from keras.models import Model

class DEC:
    def __init__(self, vae, n_clusters):
        self.encoder = vae.encoder
        self.n_clusters = n_clusters


    def cluster(self, x):
        encoded_x = self.encoder.predict(x)
        kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = kmeans.fit_predict(encoded_x)
        return clusters

if __name__ == "__main__":
    x_train = np.random.rand(1000, 64)  # Example data
    vae = VAE(input_dim=64, latent_dim=10)
    vae.train(x_train)
    dec = DEC(vae, n_clusters=10)
    clusters = dec.cluster(x_train)
    print(clusters)
