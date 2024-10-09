import os
import glob
from typing import List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer



def read_documents(folder_path: str) -> List[str]:
    '''Чтение текстов из файлов'''
    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents



def preprocess_text(text: str) -> str:
    '''Предобработка текста'''
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer() #приведения слов к корневой форме
    
    tokens = word_tokenize(text.lower()) # Токенизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words] # Удаление стоп-слов и лемматизация
    
    return ' '.join(tokens)
