import warnings

from typing import List
from numpy import ndarray
from sentence_transformers import SentenceTransformer


def embedd(processed_documents: List[str]) -> ndarray:
    '''преобразование текстов в эмбеддинги'''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Игноририе предупреждния FutureWarning 
        try:
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except Exception as e:
            print(f"An error occurred: {e}")

    embeddings = model.encode(processed_documents) # Получение эмбеддингов

    return embeddings