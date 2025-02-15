from collections import Counter, deque
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet_ic, wordnet
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import csv
import nltk


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB,
                "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)


def get_lemmatization_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence = ([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
    h = dict(nltk.pos_tag(sentence))
    stopword = stopwords.words('english')
    punct = ["(", ")", "e.g", "eg", "es", "someone", "something", "'"]
    stopword.extend(punct)
    tag = ['IN', 'DT', '.', ',', 'TO', 'CC', 'RB', 'VBP',
           'PRP']
    sent = []
    for key, value in list(h.items()):  # Eliminazione dei tag che non ci servono
        if (value not in tag) and (key not in stopword):
            sent.append(key)
    return sent


def get_text_from_file(path):
    with open(path, 'r', newline='', encoding='utf-8') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        rows = list(tsvreader)
        num_columns = len(rows[0])
        columns = [[] for _ in range(1, num_columns)]
        columns_full_definition = [[] for _ in range(1, num_columns)]
        for row in rows[1:]:
            for i, cell in enumerate(row[1:], start=1):
                if i < num_columns:
                    columns[i - 1].extend(get_lemmatization_sentence(cell))
                    columns_full_definition[i - 1].append(cell)
    return columns, columns_full_definition


def get_most_freq_words(text, nword):
    genus = []
    for row in text:
        c = Counter()
        c.update(row)
        genus.append(c.most_common(nword))
    return genus


def get_genus_list(genus):
    genus_list = []
    for el in genus:
        genus_list_inner = []
        for el2 in el:
            genus_list_inner.append(el2[0])
        genus_list.append(genus_list_inner)
    return genus_list


def get_synset_lesk(genus, definitions):
    synset_list = []
    for term in genus:
        majority_list = []
        grouped_definitions = []
        index = 0
        grouping_factor = 5
        while index < len(definitions):
            grouped_definition = ""
            for iteration in range(min(len(definitions) - index, grouping_factor)):
                grouped_definition += definitions[index + iteration] + ". "
            index = index + grouping_factor
            grouped_definitions.append(grouped_definition)

        for definition in grouped_definitions:
            sense = lesk(definition, term)
            majority_list.append(sense)
        counter_names = Counter(majority_list)
        most_common_names = counter_names.most_common(1)
        most_frequent_value_name = most_common_names[0][0]
        synset_list.append(most_frequent_value_name)

    return synset_list


def find_hyponyms_within_depth_limit(syn, max_depth):
    # Initialize a queue for breadth-first search
    if syn is None:
        return []
    queue = deque([(syn, 0)])  # Each element is a tuple (synset, depth)

    # Initialize a set to store hyponyms
    hyponyms = set()

    # Perform breadth-first search with depth limit
    while queue:
        current_syn, depth = queue.popleft()
        if depth <= max_depth:
            hyponyms.update(current_syn.hyponyms())
            for hyponym in current_syn.hyponyms():
                queue.append((hyponym, depth + 1))

    # Return the list of hyponyms
    return list(hyponyms)


def get_synset(word):
    if (len(wn.synsets(word)) > 0):
        return wn.synsets(word)[0]
    return None


def approccio1(num_genus, max_depth_hyponims, num_most_freq_word, starting_words, choosen_synsets_by_us, genus_list,
               genus, descriptions):
    synset_genus_list = []
    for genus_index in range(len(genus_list)):
        synset_genus_list.append(get_synset_lesk(genus_list[genus_index], descriptions[genus_index]))

    result_list = []
    for genus_index in range(len(synset_genus_list)):
        genus_synsets = synset_genus_list[genus_index]
        set_of_hyponims = set(genus_synsets.copy())
        for synset in genus_synsets:
            set_of_hyponims = set_of_hyponims.union(set(find_hyponyms_within_depth_limit(synset, max_depth_hyponims)))

        list_of_hyponims = list(set_of_hyponims)
        score_list = []

        for hyponim in list_of_hyponims:
            similarity_score = 0
            total_count = 0
            for synset in genus_synsets:
                for pair in genus[genus_index]:
                    # if hyponim.name() != synset.name():
                    total_count += pair[1]
                    # similarity_score += pair[1] * hyponim.wup_similarity(synset)  # Wu-Palmer Similarity, alternative to path_similarity
                    similarity_score += pair[1] * synset.path_similarity(hyponim)
            similarity_score /= total_count
            score_list.append((hyponim, similarity_score))
        score_list.sort(key=lambda tup: tup[1], reverse=True)
        result_list.append(score_list[0][0])
    return result_list, synset_genus_list


def approccio2(num_genus, max_depth_hyponims, num_most_freq_word, starting_words, choosen_synsets_by_us, genus_list,
               genus, descriptions):
    result_list = []
    for genus_index in range(len(genus_list)):
        genus_synsets = []
        for word in genus_list[genus_index]:
            for synset in wn.synsets(word):
                genus_synsets.append(synset)

        set_of_hyponims = set(genus_synsets.copy())
        for synset in genus_synsets:
            set_of_hyponims = set_of_hyponims.union(set(find_hyponyms_within_depth_limit(synset, max_depth_hyponims)))

        list_of_hyponims = list(set_of_hyponims)
        score_list = []
        for hyponim in list_of_hyponims:
            tokenized_hypo_definition = word_tokenize(hyponim.definition())
            stemmer = PorterStemmer()
            # tokenized_hypo_definition = [stemmer.stem(word) for word in tokenized_hypo_definition]
            score = 0
            for pair in genus[genus_index]:
                score += tokenized_hypo_definition.count(pair[0]) * pair[1]
            score_list.append((hyponim, score))
        score_list.sort(key=lambda tup: tup[1], reverse=True)
        result_list.append(score_list[0][0])
    return result_list


def main():
    starting_words = ["Door", "Ladybug", "Pain", "Blurriness"]
    choosen_synsets_by_us = [wordnet.synset('door.n.01'), wordnet.synset('ladybug.n.01'), wordnet.synset('pain.n.03'),
                             wordnet.synset('indistinctness.n.01')]

    file, descriptions = get_text_from_file('data/TLN-definitions-23.tsv')

    # Approccio 1 -------------------------------------------------------------------------------------------
    num_genus = 3
    num_most_freq_word = 15
    max_depth_hyponims = 10
    genus = get_most_freq_words(file, num_genus)
    genus_list = get_genus_list(genus)
    print(
        "Approccio 1 \n "
        "------------------------------------------------------------------------------------------------------------------------------")
    result_list, synset_genus_list = approccio1(num_genus, max_depth_hyponims, num_most_freq_word, starting_words,
                                                choosen_synsets_by_us, genus_list, genus, descriptions)
    key_words_defs = get_most_freq_words(file, num_most_freq_word)
    for i in range(len(starting_words)):
        print("Word to discover: " + starting_words[i])
        print("Synset choosen by us: " + str(choosen_synsets_by_us[i]))
        print("Most frequent words: " + str(key_words_defs[i]))
        print("Genus: " + str(genus_list[i]))
        print("Genus Synsets: " + str(synset_genus_list[i]))
        print("Predicted Synset: " + str(result_list[i]))
        print("Similarity score between Choosen Synset and Predict Synset: " + str(
            choosen_synsets_by_us[i].path_similarity(result_list[i])))
        print("\n")
    # ---------------------------------------------------------------------------------------------------------

    # Approccio 2 -------------------------------------------------------------------------------------------
    num_genus = 15
    num_most_freq_word = 15
    max_depth_hyponims = 15
    genus = get_most_freq_words(file, num_genus)
    stemmed_genus = []
    '''stemmer = PorterStemmer()
    for i in range(len(genus)):
        stemmed_genus.append([])
        for j in range(len(genus[i])):
            stemmed_genus[i].append((stemmer.stem(genus[i][j][0]), genus[i][j][1]))
    genus_list = stemmed_genus'''
    genus_list = get_genus_list(genus)

    print(
        "Approccio 2 \n "
        "------------------------------------------------------------------------------------------------------------------------------")
    result_list = approccio2(num_genus, max_depth_hyponims, num_most_freq_word, starting_words,
                             choosen_synsets_by_us, genus_list, genus, descriptions)
    key_words_defs = get_most_freq_words(file, num_most_freq_word)
    for i in range(len(starting_words)):
        print("Word to discover: " + starting_words[i])
        print("Synset choosen by us: " + str(choosen_synsets_by_us[i]))
        print("Most frequent words: " + str(key_words_defs[i]))
        print("Genus: " + str(genus_list[i]))
        print("Predicted Synset: " + str(result_list[i]))
        print("Similarity score between Choosen Synset and Predict Synset: " + str(
            choosen_synsets_by_us[i].path_similarity(result_list[i])))
        print("\n")
    # ---------------------------------------------------------------------------------------------------------
    # Approccio 1 è più veloce ma meno preciso dell'approccio 2


if __name__ == '__main__':
    main()
