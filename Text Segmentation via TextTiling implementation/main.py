import math
import re
import numpy as np
import pandas as pd

import nltk.tokenize.texttiling
from matplotlib import pyplot as plt
from nltk import word_tokenize, FreqDist, PorterStemmer
from nltk.corpus import stopwords

from sklearn.metrics.pairwise import cosine_similarity


def demo():
    tt = nltk.TextTilingTokenizer()
    text = file_to_string("../File_test_4a_2.txt")
    blocchi = tt.tokenize(text)
    print(blocchi)


def filter_values(values, threshold):
    to_return = []
    indices = []
    for i in range(values.__len__()):
        if values[i][1] > threshold:
            to_return.append([values[i][0], values[i][1]])
            indices.append(values[i][0])
    return to_return, indices


def depth_score_array(block_similarity_vec):
    to_return = []
    only_values = []
    if not block_similarity_vec.__len__() > 1:
        raise RuntimeError("not block_similarity_vec.__len__() > 1")
    # assert block_similarity_vec.__len__() > 1
    for i in range(block_similarity_vec.__len__()):
        depth_score = block_similarity_vec[i][1]
        if 0 < i < block_similarity_vec.__len__() - 1:
            depth_score = (block_similarity_vec[i - 1][1] - block_similarity_vec[i][1]) + (
                    block_similarity_vec[i + 1][1] - block_similarity_vec[i][1])
        elif i == 0:
            depth_score = (block_similarity_vec[i + 1][1] - block_similarity_vec[i][1])
        else:
            depth_score = (block_similarity_vec[i - 1][1] - block_similarity_vec[i][1])
        to_return.append([i, depth_score])
        only_values.append(depth_score)
    return to_return, only_values


def array_smoothing(array, size):
    to_return = []
    only_values = []
    for i in range(array.__len__()):
        average = array[i]
        actual_size = 1
        for j in range(int(size / 2) + 1):
            if j > 0:
                if i - j >= 0:
                    average += array[i - j] / 2
                    actual_size += 1
                if i + j < array.__len__():
                    average += array[i + j] / 2
                    actual_size += 1
        average /= actual_size
        to_return.append([i, average])
        only_values.append(average)
    return to_return, only_values


def calculate_block_similarity_array(blocks_vecs):
    similarity_array = []
    for i in range(blocks_vecs.__len__()):
        if i > 0:
            array = np.array(blocks_vecs[i - 1]).reshape(1, -1)
            next_array = np.array(blocks_vecs[i]).reshape(1, -1)
            similarity_array.append([i, cosine_similarity(array, next_array)[0][0]])
        else:
            similarity_array.append([i, 0.5])
    return similarity_array


def extract_blocks_vecs(freq_matrix, block_ends):
    vecs = []
    last_end = 0
    for i in range(block_ends.__len__()):
        end = block_ends[i]
        vec = freq_matrix[freq_matrix.columns[last_end]]
        if i < block_ends.__len__() - 1:
            for j in range(last_end + 1, end):
                vec += freq_matrix[freq_matrix.columns[j]]
        else:
            for j in range(last_end + 1, freq_matrix.columns.__len__()):
                vec += freq_matrix[freq_matrix.columns[j]]
        vecs.append(vec)
        last_end = end
    # bool totest = (added_vectors == freq_matrix.columns.__len__())
    return vecs


def create_initial_blocks(phrases, block_len):
    blocks_ends = []
    for i in range(math.ceil(phrases.__len__() / block_len)):
        i += 1
        if i * block_len < phrases.__len__():
            blocks_ends.append(i * block_len)
        else:
            blocks_ends.append(phrases.__len__())
            break
    return blocks_ends


def calculate_frequency_matrix(phrases, dictionary):
    dataframe = []
    dictionary_counter = extract_word_frequency(dictionary)
    counter_phrase = 0
    for phrase in phrases:
        counter_phrase += 1
        counter = extract_word_frequency(phrase)
        missing_columns = set(dictionary) - set(phrase)
        for col in missing_columns:
            counter[col] = 0

        if counter_phrase > 1:
            column_to_add = pd.Series(counter).sort_index()
            dataframe = pd.concat([dataframe, column_to_add.rename(counter_phrase)], axis=1)
            # dataframe[str(counter_phrase)] = column_to_add

        else:
            to_concat = pd.DataFrame.from_dict(counter, orient='index', columns=[str(counter_phrase)]).sort_index()
            dataframe = to_concat
    return dataframe


def dictionary_extraction(document):
    tokens = word_tokenize(document)
    fdist = FreqDist(tokens)
    word_array = list(fdist.keys())
    return word_array


def extract_word_frequency(document_tokens):
    fdist = FreqDist(document_tokens)
    word_frequency = dict(fdist)
    return word_frequency


def subdivision_pseudo_phrases(document, len_pseudo_phrase):
    remaining_tokens = word_tokenize(document)
    token_counts = remaining_tokens.__len__()
    to_return = []
    for i in range(math.ceil(token_counts / len_pseudo_phrase)):
        to_add = []
        for j in range(len_pseudo_phrase):
            if remaining_tokens.__len__() > 0:
                to_add.append(remaining_tokens.pop(0))
        to_return.append(to_add)
    return to_return


def remove_tokens_and_return_index(tokens):
    modified_tokens = []
    removed_indices = []
    for index in range(tokens.__len__()):
        if re.match(r'[^\w\s]', tokens[index]) or re.match(r'\d', tokens[index]) or re.match(r'[()]', tokens[index]):
            removed_indices.append(index)
        else:
            modified_tokens.append(tokens[index])
    return modified_tokens, removed_indices


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens, removed_index_regex = remove_tokens_and_return_index(word_tokenize(text))
    filtered_tokens = [token for i, token in enumerate(tokens) if token.lower() not in stop_words]
    removed_indices = [i for i, token in enumerate(tokens) if token.lower() in stop_words]
    removed_indices = sorted(removed_indices + removed_index_regex)
    return ' '.join(filtered_tokens), removed_indices


def perform_stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_text = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_text)


def file_to_string(path):
    with open(path, 'r', encoding="utf8") as file:
        file_string = file.read()
    return file_string


# define a function that return a graph plot line of an array:
def plot_graph(array, x_label, y_label, title):
    import matplotlib.pyplot as plt
    plt.plot(array)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()  # show the plot on the screen


# plot a histogram of an array:
def plot_bar(array, x_label, y_label, title, name):
    plt.clf()
    plt.bar(range(len(array)), array)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Save the plot as a file in folder plots/hist/:
    plt.savefig('plots/hist/' + str(name) + '.png')
    plt.show()


def plot_arrays(x, y, z, x_label, y_label, title, name):
    plt.clf()

    # Plotting the array y
    plt.plot(y, label='original')

    # Plotting the array x
    plt.plot(x, label='smoothed')

    first = True
    for values in z:
        if first:
            plt.axvline(values, color='red', linestyle='--', label='EndOfBlock')
            first = False
        else:
            plt.axvline(values, color='red', linestyle='--')

    # Adding labels to the x and y axes
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Adding a title to the plot
    plt.title(title)

    # Add a blue dot at x axis in the 5th position in the plot
    # plt.plot(x, y, 'bo')

    # Adding a legend
    plt.legend()

    # Showing the plot
    plt.savefig('plots/line/' + str(name) + '.png')
    plt.show()


def apply_lowpass_filter(data, window_size):
    indexed = []
    not_indexed = []
    window = np.ones(window_size) / window_size
    not_indexed = np.convolve(data, window, mode='same')
    for i in range(not_indexed.__len__()):
        indexed.append([i, not_indexed[i]])
    return indexed, not_indexed


def indexed_array(array):
    indexed = []
    for i in range(array.__len__()):
        indexed.append([i, array[i]])
    return indexed


def remove_index(array):
    not_indexed = []
    for i in range(array.__len__()):
        not_indexed.append(array[i][1])
    return not_indexed


def calculate_block_ends_after_filter(prev_block_ends, filtered_index):
    if not prev_block_ends.__len__() >= filtered_index.__len__():
        raise RuntimeError("not prev_block_ends.__len__() >= filtered_index.__len__()")
    # assert prev_block_ends.__len__() >= filtered_index.__len__()

    block_ends = []
    for i in range(filtered_index.__len__()):
        if (i + 1) < filtered_index.__len__():
            last_sentence = prev_block_ends[filtered_index[i + 1] - 1]
        else:
            last_sentence = prev_block_ends[prev_block_ends.__len__() - 1]
        block_ends.append(last_sentence)
    return block_ends


def calculate_block_size(block_ends):
    if not block_ends.__len__() > 1:
        raise RuntimeError("not block_ends.__len__() > 1")
    # assert block_ends.__len__() > 1
    block_size = [block_ends[0]]
    for i in range(block_ends.__len__() - 1):
        block_size.append(block_ends[i + 1] - block_ends[i])
    return block_size


def find_right_block_ends(text, special_token, sentence_size):
    tokens = word_tokenize(text)
    block_ends = []
    for i in range(tokens.__len__()):
        if tokens[i] == special_token:
            block_ends.append(math.ceil(i / sentence_size))
    return block_ends


def subdivideTextBlockEnds(text, block_ends, removed_indices, sentence_size):
    to_start = 0
    tokens = word_tokenize(text)
    to_return = []
    block_size = calculate_block_size(block_ends)
    for i in range(block_ends.__len__()):
        end_of_sentences = to_start
        removed = 0
        added = 0
        while added - removed < block_size[i] * sentence_size:
            end_of_sentences += 1
            added += 1
            if removed_indices.__contains__(end_of_sentences):
                removed += 1
        token_to_concat = []
        last_token = end_of_sentences
        if last_token > tokens.__len__():
            last_token = tokens.__len__()
        for j in range(to_start, last_token):
            token_to_concat.append(tokens[j])
        to_return.append(' '.join(token_to_concat))
        to_start = last_token
    return to_return


def find_block_index_of_sentence(block_ends, sentence_index):
    for i in range(block_ends.__len__()):
        if block_ends[i] > sentence_index:
            return i - 1
    return block_ends.__len__() - 1


def calculateAdjustedBlockSizes(text, block_ends, removed_index, sentence_size):
    text_subdivisions = subdivideTextBlockEnds(text, block_ends, removed_index, sentence_size)
    to_return = []
    for subdivision in text_subdivisions:
        to_return.append(word_tokenize(subdivision).__len__())
    return to_return


def calculateAccuracyAndRecall(block_ends, true_block_ends, number_of_sentences):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for first_sentence_index in range(number_of_sentences):
        for second_sentence_index in range(first_sentence_index + 1, number_of_sentences):
            are_in_same_block = False
            if find_block_index_of_sentence(true_block_ends, first_sentence_index) == find_block_index_of_sentence(
                    true_block_ends, second_sentence_index):
                are_in_same_block = True
            is_calculated_as_same_block = False
            if find_block_index_of_sentence(block_ends, first_sentence_index) == find_block_index_of_sentence(
                    block_ends, second_sentence_index):
                is_calculated_as_same_block = True
            if are_in_same_block and is_calculated_as_same_block:
                tp += 1
            elif are_in_same_block and not is_calculated_as_same_block:
                fn += 1
            elif not are_in_same_block and is_calculated_as_same_block:
                fp += 1
            elif not are_in_same_block and not is_calculated_as_same_block:
                tn += 1
    accuracy = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, recall


def TextTilingIteration(freq_matrix, block_ends, text, removed_index, with_filter=True, paper_filter=True, iteration=0,
                        w=0,
                        k=0, metric="LC"):
    block_vecs = extract_blocks_vecs(freq_matrix, block_ends)
    block_similarity_array = calculate_block_similarity_array(block_vecs)
    depth_array, depth_values = depth_score_array(block_similarity_array)
    # plot_arrays(depth_values, remove_index(block_similarity_array), [])
    title = "Text Tiling iteration n:" + str(iteration) + " with parameters: cut-off=HC, w=" + str(w) + ", k=" + str(k)
    # plot_graph(remove_index(block_similarity_array), "Block number", "Similarity-score", title)

    if with_filter:
        if not paper_filter:
            depth_array_smooth, depth_values_smooth = apply_lowpass_filter(depth_values, 2)
        else:
            title += ", s=2"
            depth_array_smooth, depth_values_smooth = array_smoothing(depth_values, 2)
        sd_depth_array = np.array(depth_values_smooth).var()
        mean_depth_array = np.array(depth_values_smooth).mean()
        if metric == "HC":
            sd_depth_array = sd_depth_array/2
        filtered_depth_array, indices = filter_values(depth_array_smooth,
                                                      mean_depth_array - sd_depth_array)
        # plot_arrays(depth_values_smooth, depth_values, indices, "Block number", "Depth-score", title, iteration)
    else:
        sd_depth_array = np.array(depth_values).var()
        mean_depth_array = np.array(depth_values).mean()
        if metric == "HC":
            sd_depth_array = sd_depth_array/2
        filtered_depth_array, indices = filter_values(depth_array,
                                                      mean_depth_array - sd_depth_array)
        # plot_arrays(depth_values, depth_values, indices, "Block number", "Depth-score", title, iteration)
    to_return = calculate_block_ends_after_filter(block_ends, indices)
    adjusted_block_sizes = calculateAdjustedBlockSizes(text, to_return, removed_index, w)
    adjusted_block_sizes = [math.floor(x / w) for x in adjusted_block_sizes]
    # plot_bar(adjusted_block_sizes, "Block number", "Size (pseudo-phrases)", title, iteration)
    return to_return


def text_tiling(block_size, sentence_size, iterations, text, matrix, block_ends, removed_index, sentences, metric="LC"):
    for i in range(iterations):
        block_ends = TextTilingIteration(matrix, block_ends, text, removed_index, True, True, i + 1,
                                         sentence_size, block_size, metric)
        # print(block_ends)

    accuracy, recall = calculateAccuracyAndRecall(block_ends,
                                                  find_right_block_ends(text, '§', sentence_size),
                                                  sentences.__len__())
    return accuracy, recall


def main():
    block_size = 2
    sentence_size = 10
    iterations = 10

    testo_originale = file_to_string("wikipedia_page.txt")
    testo, indici_token_rimossi = remove_stopwords(testo_originale)
    testo = perform_stemming(testo)
    dizionario = dictionary_extraction(testo)
    frasi = subdivision_pseudo_phrases(testo, sentence_size)
    matrix = calculate_frequency_matrix(frasi, dizionario)
    block_ends = create_initial_blocks(frasi, block_size)

    accuracy, recall = text_tiling(block_size, sentence_size, iterations, testo_originale, matrix, block_ends,
                                   indici_token_rimossi, frasi, metric="HC")
    print("Accuracy:" + str(accuracy), "Recall:" + str(recall),
          "F1-score:" + str(2 * ((accuracy * recall) / (accuracy + recall))))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# iterate_text_tiling()
# demo()


"""
def iterate_text_tiling():
    pair_list = []
    combined_list = []
    accuracy_list = []
    recall_list = []
    sentence_size = 10

    text = file_to_string("wikipedia_page.txt")
    testo, indici_token_rimossi = remove_stopwords(text)
    testo = perform_stemming(testo)
    dizionario = dictionary_extraction(testo)
    frasi = subdivision_pseudo_phrases(testo, sentence_size)
    matrix = calculate_frequency_matrix(frasi, dizionario)

    for block_size in range(1, 20):
        block_ends = create_initial_blocks(frasi, block_size)
        for iteration in range(1, 20):
            print("(" + str(block_size) + ", " + str(iteration) + ")")
            try:
                accuracy, recall = text_tiling(block_size, sentence_size, iteration, text,
                                               matrix, block_ends, indici_token_rimossi, frasi)
                accuracy_list.append(accuracy)
                recall_list.append(recall)
                combined_list.append(2 * ((accuracy * recall) / (accuracy + recall)))
                pair_list.append((block_size, iteration, accuracy, recall, 2 * ((accuracy * recall) / (accuracy + recall))))

                print("Accuracy: " + str(accuracy_list))
                print("Recall: " + str(recall_list))
                print("Combined: " + str(combined_list))
                print("Pair: " + str(pair_list))
                print("OK")
            except:
                print("Error in: " + str(block_size) + " " + str(iteration))
    print("Accuracy: " + str(accuracy_list))
    print("Recall: " + str(recall_list))
    print("Combined: " + str(combined_list))
    print("Pair: " + str(pair_list))
    print("OK Var/2")
"""
