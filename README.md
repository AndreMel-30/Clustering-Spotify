# Spotify Tracks Clustering Analysis

Questo progetto applica tecniche di **Machine Learning non supervisionato** per analizzare un dataset di brani Spotify. L'obiettivo è esplorare alcune caratteristiche audio (come *danceability*, *energy*, *acousticness*) e raggruppare i brani in cluster significativi utilizzando diversi algoritmi.

## Caratteristiche del Progetto

Il codice esegue una pipeline completa di Data Science:
1.  **Data Cleaning**: Gestione valori nulli, duplicati e rimozione outlier tramite metodo IQR.
2.  **Feature Scaling**: Normalizzazione dei dati con `MinMaxScaler`.
3.  **Riduzione Dimensionale**:
    * **PCA** (Principal Component Analysis) per ridurre la complessità.
    * **t-SNE** per la visualizzazione 2D di dati ad alta dimensionalità.
4.  **Clustering**:
    * **K-Means**: Ottimizzazione del K tramite *Elbow Method* e *Silhouette Score*.
    * **DBSCAN**: Clustering basato sulla densità per gestire il rumore.
    * **HDBSCAN**: Clustering gerarchico basato sulla densità.
5.  **Visualizzazione**: Grafici interattivi 2D e 3D con `Matplotlib`.

## Tecnologie Utilizzate

* **Python 3.x**
* **Pandas & NumPy**: Manipolazione dati.
* **Scikit-Learn**: Algoritmi di ML (KMeans, DBSCAN, PCA, t-SNE).
* **HDBSCAN**: Algoritmo di clustering avanzato.
* **Matplotlib**: Visualizzazione dati.

## Installazione

1.  **Clona il repository**:
    ```bash
    git clone [https://github.com/tuo-username/spotify-clustering.git](https://github.com/tuo-username/spotify-clustering.git)
    cd spotify-clustering
    ```

2.  **Installa le dipendenze**:
    Si consiglia di utilizzare un ambiente virtuale.
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

Il progetto utilizza lo **Spotify Tracks Dataset** (di Maharshi Pandya) disponibile su Kaggle.

**Come aggiungere i dati:**
1.  Scarica il dataset da [questo link Kaggle](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset).
2.  Posiziona il file scaricato `spotify-tracks-dataset.zip` nella cartella principale del progetto.
3.  Lo script estrarrà automaticamente il file CSV all'avvio.

*Alternativa*: Se hai configurato le API di Kaggle, puoi decommentare le righe relative al download automatico all'inizio dello script `clustering_analysis.py`.

## Utilizzo

Esegui lo script principale da terminale:

```bash
python clustering_analysis.py

## Risultati e Visualizzazioni

Di seguito sono riportati i grafici generati dalla pipeline di analisi.

### 1. Riduzione Dimensionale
Visualizzazione preliminare del dataset proiettato in 2D utilizzando PCA e t-SNE.
![t-SNE Projection](images/tsne_pca.png)

### 2. K-Means Analysis
Determinazione del numero ottimale di cluster e caratterizzazione dei gruppi.

| Elbow Method | Analisi delle Features (Medie) |
| :---: | :---: |
| ![Elbow Method](images/elbow_method.png) | ![Features per Cluster](images/cluster_features.png) |

**Visualizzazione 3D dei Cluster (K-Means):**
![K-Means 3D Plot](images/kmeans_3d.png)

### 3. Clustering basato sulla densità (DBSCAN & HDBSCAN)
Confronto con algoritmi che gestiscono il rumore e forme non sferiche.
![DBSCAN Result](images/dbscan_3d.png)
