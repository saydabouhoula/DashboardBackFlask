import psycopg2
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Établir la connexion à la base de données
conn = psycopg2.connect(
    dbname="DWH",
    user="postgres",
    password="sayda",
    host="localhost",
    port="5432"
)

# Définir la requête SQL
sql = """
SELECT date_pk, group_pk, product_pk, mp_pk, "Humidite", "Aw", "Proteine", "Amidon", "Fibre", "Calcium", "Probleme"
FROM public."Fait";
"""

# Lire les données dans un DataFrame Pandas
df = pd.read_sql_query(sql, conn)

# Fermer la connexion à la base de données
conn.close()

# Convertir les valeurs en float et remplacer les virgules par des points
df["Humidite"] = df["Humidite"].str.replace(',', '.').astype(float)
df["Aw"] = df["Aw"].str.replace(',', '.').astype(float)
df["Proteine"] = df["Proteine"].str.replace(',', '.').astype(float)
df["Amidon"] = df["Amidon"].str.replace(',', '.').astype(float)
df["Fibre"] = df["Fibre"].str.replace(',', '.').astype(float)
df["Calcium"] = df["Calcium"].str.replace(',', '.').astype(float)

# Remplacer les valeurs manquantes (NaN) par la moyenne des colonnes
df.fillna(df.mean(), inplace=True)

# Sélectionner les caractéristiques et la cible
X = df[['Humidite', 'Aw', 'Proteine', 'Amidon', 'Fibre', 'Calcium']]
y = df['Probleme']  # Supposons que 'Probleme' est binaire (1 pour problème, 0 pour pas de problème)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Enregistrer le modèle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

if __name__ == '__main__':
    # Entraîner et sauvegarder le modèle
    pass
