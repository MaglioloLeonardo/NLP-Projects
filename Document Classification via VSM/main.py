import math
import os
import re
from collections import Counter
from collections import defaultdict
import bisect
import pickle
import threading

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

import spacy


def loadDocuments(folder_path):
    set_classes = set()
    documents = defaultdict(list)
    n_documents = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        current_class = filename.split("_", 1)[0]
        set_classes.add(current_class)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding="utf8") as file:
                content = file.read()
                documents[current_class].append(content)
                n_documents = n_documents + 1
    return n_documents, documents, list(set_classes)


def getSpacyDocuments(documents, list_classes, nlp):
    to_disable = []
    spacy_documents = {}

    for class_name in list_classes:
        spacy_documents[class_name] = list(
            nlp.pipe(documents[class_name], disable=to_disable, n_process=4, batch_size=2000))
    return spacy_documents


def getSpacyDocumentsMultiThread(documents, list_classes, nlp):
    to_disable = ['ner', 'parser']
    spacy_documents = {}
    list_of_threads = []

    class spacyThread(threading.Thread):
        def __init__(self, class_name, documents, spacy_documents):
            threading.Thread.__init__(self)
            self.class_name = class_name
            self.documents = documents
            self.spacy_documents = spacy_documents

        def run(self):
            self.spacy_documents[self.class_name] = list(
                nlp.pipe(self.documents[self.class_name], disable=to_disable, n_process=4, batch_size=100))

    for class_name in list_classes:
        thread = spacyThread(class_name, documents, spacy_documents)
        list_of_threads.append(thread)
        thread.start()

    for thread in list_of_threads:
        thread.join()

    return spacy_documents


def extractAbsoluteFrequency(list_classes, spacy_documents):
    word_counts = Counter()
    word_in_documents = Counter()
    list_of_documents_frequencies = []
    dictionary_documents_frequencies = defaultdict(list)

    for class_name in list_classes:
        for document in spacy_documents[class_name]:
            tokens = [token.lemma_ for token in document if
                      not token.is_stop and not token.like_num and token.is_alpha and not token.pos_ in ["DET",
                                                                                                         "CCONJ"]]
            word_counts.update(tokens)
            word_in_documents.update(list(set(tokens)))
            vector = Counter()
            vector.update(tokens)
            list_of_documents_frequencies.append(vector)
            dictionary_documents_frequencies[class_name].append(vector)
    return word_counts, word_in_documents, dictionary_documents_frequencies,


def IDFCorrection(n_documents, list_classes, dictionary, word_in_documents, frequencies):
    idf_dictionary = {}
    for entry in dictionary:
        idf_dictionary[entry] = math.log2(n_documents / word_in_documents[entry])

    for class_name in list_classes:
        for vector in frequencies[class_name]:
            for entry in vector.keys():
                vector[entry] = vector[entry] * idf_dictionary[entry]

    return idf_dictionary


def conversionInDataframe(dictionary, embeddings, list_classes):
    new_dictionary = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Count'])
    df_embeddings = {}
    count = 0
    for class_name in list_classes:
        new_embeddings = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Count']).transpose()
        new_embeddings = new_embeddings.drop(new_embeddings.index[0])
        for embedding in embeddings[class_name]:
            missing_columns = set(new_embeddings.columns) - set(embedding.keys())
            for col in missing_columns:
                embedding[col] = 0
            to_concat = pd.DataFrame.from_dict(embedding, orient='index', columns=['Count']).transpose()
            new_embeddings = pd.concat([new_embeddings, to_concat], axis=0)
            count = count + 1
            print(count)
        df_embeddings[class_name] = new_embeddings
    return new_dictionary, df_embeddings


def computeEmbeddingsAndSerialize(folder_path):
    n_documents, documents, list_classes = loadDocuments(folder_path)
    spacy_documents = getSpacyDocuments(documents, list_classes, spacy.load("it_core_news_lg"))
    dictionary, word_in_documents, frequencies = extractAbsoluteFrequency(list_classes, spacy_documents)
    IDFCorrection(n_documents, list_classes, dictionary, word_in_documents, frequencies)
    dictionary, embeddings = conversionInDataframe(dictionary, frequencies, list_classes)

    dictionary.to_csv(os.getcwd() + '/dataframes/dictionary.csv', index=True)
    for class_name in list_classes:
        embeddings[class_name].to_csv(os.getcwd() + '/dataframes/df' + '_' + class_name + '_embeddings.csv', index=True)

    with open(os.getcwd() + '/dataframes/classes', 'wb') as file:
        pickle.dump(list_classes, file)
    return dictionary, embeddings, list_classes


def loadEmbeddingsAndDictionary():
    embeddings = {}
    pattern = r"df_.+_embeddings\.csv"
    for filename in os.listdir(os.getcwd() + "/dataframes"):
        file_path = os.path.join("", filename)
        if re.match(pattern, filename):
            current_class = filename.split("df_", 1)[1].split("_", 1)[0]
            if os.path.isfile(os.getcwd() + "/dataframes/" + file_path):
                df = pd.read_csv(os.getcwd() + "/dataframes/" + file_path)
                embeddings[current_class] = df.drop(df.columns[0], axis=1)

    dictionary = pd.read_csv(os.getcwd() + "/dataframes/dictionary.csv")
    with open(os.getcwd() + "/dataframes/classes", 'rb') as file:
        classes = pickle.load(file)

    return dictionary, embeddings, classes


def computeCentroids(embeddings, classes_list):
    centroids = {}
    for class_name in classes_list:
        centroids[class_name] = embeddings[class_name].mean(axis=0)
    return centroids


def computeClassEmbeddings(embeddings, classes_list, pos_coeff, neg_coeff, n_negative_examples):
    centroids = computeCentroids(embeddings, classes_list)
    to_return = {}
    for class_name in classes_list:
        negative_examples = []
        negative_similarity = []
        "calculate N most near negative examples"
        for neg_class_name in classes_list:
            if not neg_class_name == class_name:
                for index, row in embeddings[neg_class_name].iterrows():
                    similarity = \
                        cosine_similarity(centroids[class_name].values.reshape(1, -1), row.values.reshape(1, -1))[0][0]
                    if negative_similarity.__len__() > 0:
                        if similarity > negative_similarity[0]:
                            if negative_examples.__len__() >= n_negative_examples:
                                negative_examples.pop(0)
                                negative_similarity.pop(0)
                            bisect.insort(negative_similarity, similarity)
                            index_of_insertion = negative_similarity.index(similarity)
                            negative_examples.insert(index_of_insertion, row)
                    else:
                        negative_similarity.append(similarity)
                        negative_examples.append(row)
        df_negative_examples = pd.concat(negative_examples, axis=0)
        df_negative_centroid = df_negative_examples.mean(axis=0)
        to_return[class_name] = pos_coeff * centroids[class_name] - (neg_coeff * df_negative_centroid)
    return to_return


def predictClass(entry_serie, classes_embeddings, classes):
    predicted_label = ""
    prev_val = -1
    for class_name_centroid in classes:
        similarity = cosine_similarity(entry_serie.values.reshape(1, -1),
                                       classes_embeddings[class_name_centroid].values.reshape(1, -1))[0][0]
        if prev_val < similarity:
            predicted_label = class_name_centroid
            prev_val = similarity
    return predicted_label


def exportConfusionMatrixToImage(confusion_matrix_data, accuracy, path):
    # Create a 2D array from the confusion matrix data
    confusion_matrix_array = pd.crosstab(confusion_matrix_data['Actual'], confusion_matrix_data['Predicted']).values

    confusion_matrix_data['Predicted'] = confusion_matrix_data['Predicted'].str[:4]
    confusion_matrix_data['Actual'] = confusion_matrix_data['Actual'].str[:9]

    # Plot the confusion matrix
    plt.clf()
    plt.imshow(confusion_matrix_array, cmap='Blues')
    plt.title('Confusion Matrix with accuracy: ' + str(accuracy))
    plt.colorbar()
    plt.xticks(np.arange(len(confusion_matrix_data['Predicted'].unique())), confusion_matrix_data['Predicted'].unique())
    plt.yticks(np.arange(len(confusion_matrix_data['Actual'].unique())), confusion_matrix_data['Actual'].unique())
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Add text annotations to the plot
    for i in range(len(confusion_matrix_data['Actual'].unique())):
        for j in range(len(confusion_matrix_data['Predicted'].unique())):
            plt.text(j, i, confusion_matrix_array[i, j], ha='center', va='center', color='black')

    # Save the plot as an image file
    plt.savefig(path)


def crossValidation(number_of_partitions, embeddings, classes, pos_coeff, neg_coeff, n_negative_examples):
    cm_columns = ["Predicted " + classes_name for classes_name in classes]
    cm_index = ["Actual " + classes_name for classes_name in classes]
    accuracy = {}
    avg_errors = {}

    for class_name in classes:
        avg_errors[class_name] = 0

    for i in range(number_of_partitions):
        training_embeddings = {}
        test_embeddings = {}
        true_labels = []
        predicted_labels = []
        for class_name in classes:
            training_embeddings[class_name], test_embeddings[class_name] = train_test_split(embeddings[class_name],
                                                                                            test_size=0.1,
                                                                                            random_state=i)
        classes_embeddings_train = computeClassEmbeddings(training_embeddings, classes, pos_coeff, neg_coeff,
                                                          n_negative_examples)
        for class_name in classes:
            for index, row in test_embeddings[class_name].iterrows():
                predicted_label = predictClass(row, classes_embeddings_train, classes)
                predicted_labels.append(predicted_label)
                true_labels.append(class_name)
                if not predicted_label == class_name:
                    avg_errors[class_name] = avg_errors[class_name] + 1 / number_of_partitions

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        dataframe_cm = pd.DataFrame(conf_matrix, columns=cm_columns, index=cm_index)
        accuracy[i] = np.trace(conf_matrix) / np.sum(conf_matrix)
        dataframe_cm.to_csv(os.getcwd() + "/confusion_matrices/" + str(i) + ".csv", index=True)
        confusion_matrix_data = pd.DataFrame({
            'Predicted': predicted_labels,
            'Actual': true_labels
        })
        exportConfusionMatrixToImage(confusion_matrix_data, float(accuracy[i]),
                                     os.getcwd() + "/confusion_matrices/" + str(i) + ".png")
        print(f"Iteration number:{i}")

    return accuracy, avg_errors


def searchWeightInSerie(embeddings, class_name, attribute):
    return embeddings[class_name].loc[attribute]


def main():
    """dictionary, embeddings, classes = computeEmbeddingsAndSerialize("docs_200")"""
    dictionary, embeddings, classes = loadEmbeddingsAndDictionary()
    accuracy_arr, errors = crossValidation(10, embeddings, classes, 16, 4, 10)
    accuracy_arr = pd.Series(accuracy_arr)
    print("Avg errors:", errors)
    print("Accuracy by iteration:", accuracy_arr)
    print("Mean accuracy:", accuracy_arr.mean())
    print("Variance accuracy:", accuracy_arr.var())


if __name__ == '__main__':
    main()
