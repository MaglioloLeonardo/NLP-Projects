import math
import random
import time

import gensim.models
from gensim.corpora import Dictionary
import numpy as np
import pandas as pd
import multiprocessing as mp

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize


def extract_dataframe_randomly(n_entries, data_path, save=False, save_path=""):
    total_rows = sum(1 for line in open(data_path, encoding='utf-8'))
    skip_rows = sorted(random.sample(range(1, total_rows), total_rows - n_entries))
    df = pd.read_csv(data_path, skiprows=skip_rows, nrows=n_entries, encoding='utf-8', low_memory=False)
    if save:
        df.to_csv(save_path, index=False)
    return df


# sort list of dataframes by the length of the dataframe, bigger first
def sort_list_of_dataframes(list_of_dfs):
    list_of_dfs.sort(key=lambda x: len(x), reverse=True)
    return list_of_dfs


def filter_dataframe_by_section_names(df, n_values, k_labels):
    df = df[df['section_name'].notna()]
    df_to_return = pd.DataFrame(columns=df.columns)

    list_of_labels = []
    list_of_dfs = []
    for value in df['section_name'].unique():
        list_of_dfs.append(df[df['section_name'] == value])
    list_of_dfs = sort_list_of_dataframes(list_of_dfs)

    counter = 0
    for df_label in list_of_dfs:
        counter += 1
        if counter > k_labels:
            break
        list_of_labels.append(df_label['section_name'].unique()[0])
        if n_values <= len(df_label):
            df_to_return = pd.concat([df_to_return, df_label[:math.floor(n_values / k_labels)]])
        else:
            df_to_return = pd.concat([df_to_return, df_label[:math.floor((len(df_label) - 1) / k_labels)]])
    return df_to_return, list_of_labels


def pos_tag_with_nouns(tokens):
    tagged_words = nltk.pos_tag(tokens)
    nouns = [word for word, pos in tagged_words if pos.startswith('N')]
    return nouns


def build_model_using_gensim_lda(dataframe, n_topics, n_passes=10, n_iterations=100, n_words_per_topic=10,
                                 save=False, save_path="LDAModel"):
    # Load the LDA model from gensim library
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from gensim.utils import simple_preprocess
    from gensim.corpora import Dictionary

    # Tokenize the data
    def sent_to_words(sentences):
        for sentence in sentences:
            yield simple_preprocess(str(sentence), deacc=True)

    data_words = list(sent_to_words(dataframe['abstract']))
    # Remove from a list of words all the words that are not nouns
    data_words = [pos_tag_with_nouns(text) for text in data_words]

    # Create the Dictionary and Corpus needed for Topic Modeling
    id2word = Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    cores = mp.cpu_count()
    # Build the LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=n_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=n_passes,
                         alpha='auto',
                         per_word_topics=True)

    # doc_lda = lda_model[corpus]
    perplexity = lda_model.log_perplexity(corpus)

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    # Save the model
    if save:
        lda_model.save(save_path)
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(vis, 'lda.html')
    return lda_model, perplexity, coherence_lda


def build_matrix_models_LDA(dataset, n_labels, n_topics, test=False):
    perplexity_matrix = pd.DataFrame()  # rows: n_labels, columns: n_topics
    coherence_matrix = pd.DataFrame()  # rows: n_labels, columns: n_topics

    for n_labels in range(1, n_labels + 1):
        perplexity_list = []
        coherence_list = []
        print("Building model with " + str(n_labels) + " labels")
        df, labels = filter_dataframe_by_section_names(dataset, 500000, n_labels)
        for n_topics in range(1, n_topics + 1):
            if test and not n_topics == n_labels:
                perplexity_list.append(0)
                coherence_list.append(0)
                continue
            print("Building model with " + str(n_topics) + " topics")
            lda_model, perplexity, coherence = build_model_using_gensim_lda(df, n_topics, n_passes=10, save=False)
            perplexity_list.append(perplexity)
            coherence_list.append(coherence)
        new_row_perplexity = pd.DataFrame([perplexity_list])
        new_row_coherence = pd.DataFrame([coherence_list])
        perplexity_matrix = pd.concat([perplexity_matrix, new_row_perplexity])
        coherence_matrix = pd.concat([coherence_matrix, new_row_coherence])
    return perplexity_matrix, coherence_matrix


def normalize_value_array(array):
    min_val = np.min(np.abs(array))
    max_val = np.max(np.abs(array))

    # Normalize the array between 0 and 1
    return (np.abs(array) - min_val) / (max_val - min_val)


def compute_trade_off_values_array(perplexity_matrix, coherence_matrix):
    assert perplexity_matrix.shape == coherence_matrix.shape
    to_return = []
    for i in range(perplexity_matrix.shape[0]):
        perplexity_row = normalize_value_array(perplexity_matrix.iloc[i])
        coherence_row = normalize_value_array(coherence_matrix.iloc[i])
        trade_off_row = perplexity_row * coherence_row
        max_index = trade_off_row.idxmax()
        to_return.append((max_index, trade_off_row[max_index]))
    return to_return


def plot_arrays(x, y, z, x_label, y_label, z_label, title, name):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot the first array on the left y-axis (blue line)
    ax1.plot(np.arange(1, len(x) + 1), x, 'b', label=x_label)
    ax1.set_xlabel('Number of topics')
    ax1.set_ylabel(x_label, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot the second array on the right y-axis (green line)
    ax2.plot(np.arange(1, len(x) + 1), y, 'g', label=y_label)
    ax2.set_ylabel(y_label, color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    # Add a new entry in the legend for the vertical line
    plt.axvline(z, color='red', linestyle='--', label=z_label)

    # Add a legend for both arrays
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

    plt.title(title)
    # plt.legend()

    # Showing the plot
    plt.savefig('plots/' + str(name) + '.png')
    plt.show()


def generate_plots(perplexity_matrix, coherence_matrix):
    assert len(perplexity_matrix) == len(coherence_matrix)
    best_values = compute_trade_off_values_array(perplexity_matrix, coherence_matrix)
    for i in range(0, len(perplexity_matrix)):
        perplexity_row = perplexity_matrix.iloc[i]
        coherence_row = coherence_matrix.iloc[i]

        plot_arrays(perplexity_row, coherence_row, int(best_values[i][0]) + 1, 'Log Perplexity', 'Coherence Score',
                    "Best Trade-off with score:" + str(round(best_values[i][1], 2)),
                    'Scores of LDA models with ' + str(i + 1) + ' categories',
                    'lda_score_' + str(i + 1))


def test():
    size = 500000
    n_labels = 10
    path_to_data = "nyt_extracted_data_" + str(size) + ".csv"
    df = pd.read_csv("nyt_extracted_data_500000.csv", encoding='utf-8', low_memory=False)
    df = pd.read_csv(path_to_data, encoding='utf-8', low_memory=False)
    df, labels = filter_dataframe_by_section_names(df, size / 10, n_labels)
    perplexity_matrix, coherence_matrix = build_matrix_models_LDA(df, n_labels, n_labels, test=False)
    perplexity_matrix.to_csv("perplexity_matrix.csv", index=False)
    coherence_matrix.to_csv("coherence_matrix.csv", index=False)
    generate_plots(perplexity_matrix, coherence_matrix)
    print("okay")


def generate_plots_from_csv():
    perplexity_matrix = pd.read_csv("perplexity_matrix.csv", encoding='utf-8', low_memory=False)
    coherence_matrix = pd.read_csv("coherence_matrix.csv", encoding='utf-8', low_memory=False)
    generate_plots(perplexity_matrix, coherence_matrix)


def generate_optimal_models():
    size = 500000
    perplexity_matrix = pd.read_csv("perplexity_matrix.csv", encoding='utf-8', low_memory=False)
    coherence_matrix = pd.read_csv("coherence_matrix.csv", encoding='utf-8', low_memory=False)
    best_values = compute_trade_off_values_array(perplexity_matrix, coherence_matrix)
    for index in range(0, len(best_values)):
        n_topics = int(best_values[index][0]) + 1
        print("Building model with " + str(index + 1) + " labels and " + str(n_topics) + " topics")
        path_to_data = "nyt_extracted_data_" + str(size) + ".csv"
        df = pd.read_csv(path_to_data, encoding='utf-8', low_memory=False)
        df, labels = filter_dataframe_by_section_names(df, size / 10, index + 1)
        lda_model, perplexity, coherence = build_model_using_gensim_lda(df, n_topics, n_passes=10, save=True,
                                                                        save_path="Optimal_Models/LDAModel_" + str(
                                                                            n_topics))
        print(index, labels, lda_model.print_topics(), perplexity, coherence,
              "_______________________________________________________")


def main():
    size = 500000
    n_labels = 10
    # extract_dataframe_randomly(500000, "C:\\Users\\magli\\OneDrive\\Desktop\\nyt-metadata.csv", True,
    #                           "nyt_extracted_data_500000.csv")
    start_time = time.time()
    path_to_data = "nyt_extracted_data_" + str(size) + ".csv"
    df = pd.read_csv(path_to_data, encoding='utf-8', low_memory=False)
    df, labels = filter_dataframe_by_section_names(df, size / 10, n_labels)
    print(labels)
    model, perplexity, coherence = build_model_using_gensim_lda(df, n_labels, n_passes=10, save=True, save_path="LDAModel")
    # print time after the execution in seconds and minutes
    print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))
    print("model perplexity: ", perplexity)
    print("model coherence: ", coherence)
    print(model.print_topics())


def getDistribuitionText(model, text, dictionary):
    tokens = word_tokenize(text.lower())
    bow_vector = dictionary.doc2bow(tokens)
    return model.get_document_topics(bow_vector)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    generate_plots_from_csv()
    #main()
