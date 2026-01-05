# -*- coding: utf-8 -*-
"""
Spotify Tracks Clustering Analysis
Original logic based on: https://colab.research.google.com/drive/1qJeLegPx6jZzCFHgz6Wubil6fH_r0hTM
Refactored for GitHub.
"""

# --- IMPORTS ---
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # Scommentare se necessario

# Machine Learning & Clustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# 3D Plotting
from mpl_toolkits.mplot3d import Axes3D

# HDBScan (Assicurati di aver installato: pip install hdbscan)
import hdbscan

# --- CONFIGURAZIONE E DATASET ---

# NOTA PER L'UTENTE:
# Le seguenti righe sono specifiche per Google Colab o per il download automatico da Kaggle.
# Se esegui in locale, assicurati di avere il file 'kaggle.json' o scarica manualmente il dataset.

# from google.colab import files
# files.upload() # Carica kaggle.json qui se usi Colab

# Comandi Shell (Commentati per esecuzione locale Python)
# os.system("pip install hdbscan")
# os.system("mkdir -p ~/.kaggle")
# os.system("cp kaggle.json ~/.kaggle/")
# os.system("chmod 600 ~/.kaggle/kaggle.json")
# os.system("kaggle datasets download -d maharshipandya/-spotify-tracks-dataset")

# Gestione estrazione dataset
zip_filename = "-spotify-tracks-dataset.zip"
extract_folder = "dataset"
csv_path = os.path.join(extract_folder, 'dataset.csv')

# Se il file zip esiste, estrailo
if os.path.exists(zip_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
elif not os.path.exists(csv_path):
    print(f"Attenzione: Non trovo '{csv_path}' né '{zip_filename}'. Assicurati che il dataset sia presente.")

# --- CARICAMENTO E PULIZIA DATI ---

# Carica il file CSV in un DataFrame Pandas
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Errore: File CSV non trovato. Verifica il percorso.")
    exit()

df = df.drop_duplicates() # rimuove eventuali duplicati
print(f"Shape iniziale: {df.shape}")
print(f"Valori nulli:\n{df.isnull().sum()}")

colonne_da_eliminare = ['track_name','album_name','track_name', 'artists','key','mode','time_signature','explicit', 'speechiness','Unnamed: 0', "popularity"]


df = df.drop(columns=colonne_da_eliminare, errors='ignore')
df.to_csv('nuovo_dataset.csv', index=False)

# --- FEATURE SCALING ---

# Selezione colonne numeriche da scalare
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# istanza della classe MinMaxScaler
scaler = MinMaxScaler()
# calcolo il valore di minino e massimo di ogni colonna
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(df.info())

# Crea una copia del dataset originale per non modificarlo direttamente
df_cleaned = df.copy()

# Loop su tutte le colonne per calcolare i limiti IQR e rimuovere outlier
numeric_cols_clean = df_cleaned.select_dtypes(include=['float64']).columns

for col in numeric_cols_clean:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

print(f"Righe originali: {df.shape[0]}")
print(f"Righe dopo la pulizia: {df_cleaned.shape[0]}")
print(df_cleaned.info())

# --- RIDUZIONE DIMENSIONALE E VISUALIZZAZIONE PRELIMINARE ---

# PCA + t-SNE
pca = PCA(n_components=5, random_state=42)
df_pca = pca.fit_transform(df_cleaned)

# Campione del 20% per t-SNE
df_pca_sampled = df_pca[:int(0.2 * len(df_pca))]

tsne = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42)
df_tsne = tsne.fit_transform(df_pca_sampled)
df_tsne = pd.DataFrame(df_tsne, columns=['TSNE1', 'TSNE2'])

plt.figure(figsize=(10, 8))
plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'], alpha=0.5)
plt.xlabel("t-SNE Componente 1")
plt.ylabel("t-SNE Componente 2")
plt.title("Visualizzazione del dataset tramite PCA + t-SNE")
plt.show()

# --- FUNZIONI DI UTILITÀ ---

def valuta_clustering(data, labels):
    if len(set(labels)) > 1:  # Deve esserci più di un cluster valido
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
        return silhouette, davies_bouldin
    else:
        print("Unico cluster trovato o solo rumore.")
        return None, None

def find_optimal_k(df_scaled):
    inerzia = []
    range_k = range(1, 11)
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df_scaled)
        inerzia.append(kmeans.inertia_)
    
    plt.plot(range_k, inerzia, 'bx-')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Inerzia')
    plt.title('Elbow Method per selezionare il k ottimale')
    plt.grid()
    plt.show()

# --- K-MEANS CLUSTERING ---

# Elbow Method
find_optimal_k(df_scaled=df_cleaned)

# Plot manuale Silhouette Score (dati hardcoded dallo script originale)
silhouette_vals = [0., 0.267, 0.325, 0.259, 0.232, 0.227]
n_cluster_range = range(2, 8)
plt.plot(n_cluster_range, silhouette_vals, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('silhouette score')
plt.title('Selecting the number of clusters k using the silhouette score')
plt.grid()
plt.show()

# K-Means 
df_no_track_id = df_cleaned.drop(columns=['track_id', "track_genre"], errors='ignore')
print(df_no_track_id.info())

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(df_no_track_id)

print("k-means (su df_no_track_id):")
valuta_clustering(df_no_track_id, kmeans_labels)

# Visualizzazione 2D K-Means
pca = PCA(n_components=6)
data_2d = pca.fit_transform(df_no_track_id)

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.title("Visualizzazione dei cluster con K-means (PCA)")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.show()

# Visualizzazione 3D K-Means
pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned.select_dtypes(include=[np.number])) # Assicura solo numerici per PCA

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax.set_title("Visualizzazione dei cluster con K-means (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")
plt.colorbar(scatter)
plt.show()

# K-Means su dataset 'cleaned' completo
# Assumiamo df_cleaned sia stato trattato o le colonne stringa ignorate nel fit precedente.
df_cleaned_numeric = df_cleaned.select_dtypes(include=[np.number])
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(df_cleaned_numeric)

print("k-means (su df_cleaned numeric):")
valuta_clustering(df_cleaned_numeric, kmeans_labels)

# Visualizzazione 3D con Legenda Colori
pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned_numeric)
cluster_colors = {0: "viola", 1: "blu", 2: "verde", 3: "giallo"}

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=kmeans_labels, cmap='viridis', alpha=0.7)
ax.set_title("Visualizzazione dei cluster con K-means (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")

legend_elements = []
for cluster_num, color_name in cluster_colors.items():
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster_num} = {color_name}',
                                      markerfacecolor=plt.cm.viridis(cluster_num / 3), markersize=10))
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Mappa Cluster-Colore")
plt.show()

# --- ANALISI DEI CLUSTER ---

# Creazione DataFrame clustered
cols_for_clustering = ['duration_ms', 'danceability','energy','loudness','acousticness','instrumentalness','liveness','valence','tempo']
# Verifica che queste colonne esistano in df_no_track_id prima di creare il DF
available_cols = [c for c in cols_for_clustering if c in df_no_track_id.columns]
df_clustered = df_no_track_id[available_cols].copy()
df_clustered['Cluster'] = kmeans_labels

for i in range(4):
    print(f"Statistiche per il cluster {i}:")
    print(df_clustered[df_clustered['Cluster'] == i].describe())
    print("\n")

# Plot features
cluster_means = df_clustered.groupby('Cluster').mean()
colors = ['purple', 'blue', 'green', 'yellow']
cluster_means.T.plot(kind='bar', figsize=(12, 8), color=colors)
plt.title("Medie delle features per cluster")
plt.xlabel("Features")
plt.ylabel("Media")
plt.legend(title="Cluster")
plt.show()

# Analisi Generi
df_cleaned['Cluster'] = df_clustered['Cluster'] # Aggiungo i cluster al df originale
distribuzione = df_cleaned.groupby('Cluster')['track_genre'].value_counts(normalize=True) * 100
print(distribuzione)

for cluster in df_cleaned['Cluster'].unique():
    distribuzione_cluster = df_cleaned[df_cleaned['Cluster'] == cluster]['track_genre'].value_counts(normalize=True) * 100
    somma_percentuali = distribuzione_cluster.sum()
    print(f"\nSomma delle percentuali per il cluster {cluster}: {somma_percentuali:.2f}%")
    # print(distribuzione_cluster) # Decommentare per vedere dettaglio

# Visualizza Cluster 1
if 1 in distribuzione.index.get_level_values(0):
    distribuzione_cluster_1 = distribuzione.loc[1]
    distribuzione_cluster_1.plot(kind='bar', figsize=(10, 6))
    plt.title("Distribuzione dei generi per il Cluster 1")
    plt.xlabel("Genere musicale")
    plt.ylabel("Percentuale")
    plt.show()

# Plot Silhouette score hardcoded 
silhouette_vals_2 = [0.319,0.303,0.325,0.259,0.232,0.227]
n_cluster_2 = range(2, 8)
plt.plot(n_cluster_2, silhouette_vals_2, 'bx-')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('silhouette score')
plt.title('Silhouette score al variare del numero di cluster (k)')
plt.grid()
plt.show()

# --- DBSCAN ANALYSIS ---

print("--- Inizio analisi DBSCAN ---")
# Preparazione dati numerici per DBSCAN
df_cleaned_num = df_cleaned.select_dtypes(include=[np.number]).drop(columns=['Cluster'], errors='ignore')

# K-Dist Graph per Epsilon
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(df_cleaned_num)
distances, indices = neighbors_fit.kneighbors(df_cleaned_num)
distances = np.sort(distances[:, -1])[::-1]
plt.plot(distances)
plt.title("Grafico delle k-dist per scegliere epsilon")
plt.xlabel("Punti ordinati")
plt.ylabel(f"Distanza del {k}-esimo vicino")
plt.show()

# Applicazione DBSCAN (Primo tentativo)
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_cleaned_num)

print("DBSCAN (eps=0.2):")
if len(set(dbscan_labels)) > 1:
    valuta_clustering(df_cleaned_num[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
else:
    print("DBSCAN ha trovato solo rumore o un unico cluster.")

# Visualizzazione DBSCAN 3D
pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned_num)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=dbscan_labels, cmap='viridis', alpha=0.7)
ax.set_title("Visualizzazione dei cluster con DBSCAN (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")
plt.show()

# Visualizzazione DBSCAN 2D
pca = PCA(n_components=6)
data_2d = pca.fit_transform(df_cleaned_num)
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.title("Visualizzazione dei cluster con DBSCAN (PCA)")
plt.xlabel("Componente principale 1")
plt.ylabel("Componente principale 2")
plt.show()

# Applicazione DBSCAN (Secondo tentativo eps=0.15)
eps_value = 0.15
dbscan = DBSCAN(eps=eps_value, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_cleaned_num)

num_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"Numero di cluster trovati (eps={eps_value}): {num_clusters}")

if num_clusters > 1:
    valuta_clustering(df_cleaned_num[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
else:
    print("DBSCAN ha trovato solo rumore o un unico cluster.")

# Visualizzazione DBSCAN 3D (Secondo tentativo)
# Ricalcolo PCA 3D per sicurezza
pca = PCA(n_components=3)
data_3d = pca.fit_transform(df_cleaned_num)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=dbscan_labels, cmap='viridis', alpha=0.5)
ax.set_title(f"Clustering con DBSCAN eps={eps_value} (PCA in 3D)")
ax.set_xlabel("Componente principale 1")
ax.set_ylabel("Componente principale 2")
ax.set_zlabel("Componente principale 3")
plt.colorbar(scatter)
plt.show()

# --- HDBSCAN ANALYSIS ---

print("--- Inizio analisi HDBSCAN ---")
hdbscan_clusterer = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=50)
hdbscan_labels = hdbscan_clusterer.fit_predict(df_cleaned_num)

# Visualizzazione HDBSCAN
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
# FIX: Aggiunta virgola mancante prima di c=hdbscan_labels
scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=hdbscan_labels, cmap='viridis', alpha=0.5)
ax.set_title("Clustering con HDBSCAN")
plt.colorbar(scatter)
plt.show()
