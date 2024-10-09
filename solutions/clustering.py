from numpy import ndarray

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt



def silhouette_analysis(embeddings: ndarray):
    sil_scores = []
    K = range(2, 10)  # начинаем с k=2
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        sil_score = silhouette_score(embeddings, labels)
        sil_scores.append(sil_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(K, sil_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.show()



def clust(embeddings: ndarray):
    '''кластеризация'''

    silhouette_analysis(embeddings) #Silhouette Score

    optimal_k = 4 
    kmeans = KMeans(n_clusters=optimal_k)
    labels: ndarray = kmeans.fit_predict(embeddings) #кластеризация на заданных эмбедингах

    sil_score = silhouette_score(embeddings, labels)
    print(f'Метки кластеров: {labels}')
    print(f"Silhouette Score: {sil_score}") #значение коэффициента

    return labels, kmeans



def find_nearest_cluster(new_embedding: ndarray, kmeans: KMeans) -> int:
    '''Находит ближайший кластер для нового документа через косинусное расстояние'''
    centroids = kmeans.cluster_centers_  # Центроиды кластеров
    similarities = cosine_similarity(new_embedding, centroids)  
    nearest_cluster = similarities.argmax()  

    return nearest_cluster