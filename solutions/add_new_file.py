from numpy import ndarray
from sklearn.cluster import KMeans

from work_with_text import preprocess_text, read_documents
from embedding import embedd
from clustering import find_nearest_cluster



def classify_new_document(new_path_document: str, embeddings: ndarray, labels: ndarray, kmeans: KMeans):
    '''Классификация нового документа'''
    new_document = read_documents(new_path_document)
    processed_doc = [preprocess_text(doc) for doc in new_document]  # Предобработка нового текста
    new_embedding = embedd(processed_doc)  # Преобразование нового документа в эмбеддинг

    nearest_cluster = find_nearest_cluster(new_embedding, kmeans)
    
    print(f'Новый документ относится к кластеру: {nearest_cluster}')


def add_file(new_path_document: str, emd: ndarray, labels: ndarray, kmeans: KMeans):
    classify_new_document(new_path_document, emd, labels, kmeans)