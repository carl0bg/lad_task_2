from embedding import embedd
from clustering import clust
from work_with_text import read_documents, preprocess_text
from add_new_file import add_file





if __name__ == "__main__":

    folder_path = 'sampled_texts'

    documents = read_documents(folder_path)
    processed_documents = [preprocess_text(doc) for doc in documents]

    emd = embedd(processed_documents)

    labels, kmeans = clust(emd) #labels и обученный kmeans

    path_new_document = 'new_documents'
    add_file(path_new_document, emd, labels, kmeans)





