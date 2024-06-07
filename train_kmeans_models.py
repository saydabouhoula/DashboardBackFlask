import psycopg2
import pandas as pd
from sklearn.cluster import KMeans
import pickle
from sqlalchemy import create_engine
    

def train_kmeans_models():
    # Établir la connexion à la base de données
    conn = psycopg2.connect(
        dbname="DWH",
        user="postgres",
        password="sayda",
        host="localhost",
        port="5432"
    )

    # Définir la requête SQL pour récupérer les données
    sql_prod = """
    SELECT F.product_pk, P.name, F."Proteine", F."Amidon", F."Calcium"
    FROM public."Fait" F
    JOIN public."Dim_Product" P ON F.product_pk = P.product_pk
    WHERE F."Proteine" IS NOT NULL AND F."Amidon" IS NOT NULL AND F."Calcium" IS NOT NULL;
    """

    # Lire les données dans un DataFrame Pandas
    df = pd.read_sql_query(sql_prod, conn)

    # Remplacer les virgules par des points dans les colonnes 'Proteine', 'Amidon' et 'Calcium'
    df['Proteine'] = df['Proteine'].str.replace(',', '.').astype(float)
    df['Amidon'] = df['Amidon'].str.replace(',', '.').astype(float)
    df['Calcium'] = df['Calcium'].str.replace(',', '.').astype(float)

    # Regrouper par 'name' et calculer les moyennes
    df_grouped = df.groupby('name').mean().reset_index()

    # Entraîner les modèles KMeans
    kmeans_proteine = KMeans(n_clusters=3).fit(df_grouped[['Proteine']])
    kmeans_amidon = KMeans(n_clusters=3).fit(df_grouped[['Amidon']])
    kmeans_calcium = KMeans(n_clusters=3).fit(df_grouped[['Calcium']])

    # Sauvegarder les modèles KMeans dans un fichier modele.pkl
    with open('modele.pkl', 'wb') as model_file:
        pickle.dump(kmeans_proteine, model_file)
        pickle.dump(kmeans_amidon, model_file)
        pickle.dump(kmeans_calcium, model_file)

    return kmeans_proteine, kmeans_amidon, kmeans_calcium

if __name__ == '__main__':
    train_kmeans_models()
