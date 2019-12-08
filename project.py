# -*- coding: utf-8 -*-
from __future__ import division
import nltk
import random
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from collections import defaultdict
from wsd_code import senses, sense_instances, STOPWORDS_SET, wsd_context_features, wsd_word_features, extract_vocab_frequency, extract_vocab


_inst_cache = {}

STOPWORDS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$', '000', '1', '2', '10,' 'I', 'i', 'a', 'about', 'after', 'all', 'also', 'an', 'any',
             'are', 'as', 'at', 'and', 'be', 'being', 'because', 'been', 'but', 'by',
             'can', "'d", 'did', 'do', "don'", 'don', 'for', 'from', 'had','has', 'have', 'he',
             'her','him', 'his', 'how', 'if', 'is', 'in', 'it', 'its', "'ll", "'m", 'me',
             'more', 'my', 'n', 'no', 'not', 'of', 'on', 'one', 'or', "'re", "'s", "s",
             'said', 'say', 'says', 'she', 'so', 'some', 'such', "'t", 'than', 'that', 'the',
             'them', 'they', 'their', 'there', 'this', 'to', 'up', 'us', "'ve", 'was', 'we', 'were',
             'what', 'when', 'where', 'which', 'who', 'will', 'with', 'years', 'you',
             'your']

STOPWORDS_SET=set(STOPWORDS)

NO_STOPWORDS = []



#def acc(truth, preds, label):
#    return sum([(y == label and x == label) or (y != label and x != label) for x, y in zip(truth, preds)]) / len(truth)

def recall(truth, preds, label):
    return sum([x == label and y == label for x, y in zip(preds, truth)]) / sum([x == label for x in truth])

def precision(truth, preds, label):
    return sum([x == label and y == label for x, y in zip(preds, truth)]) / sum([x == label for x in preds])

def accuracy_per_label(truth, preds, label):
    return sum([(x == label and y == label) or (x != label and y != label) for x, y in zip(preds, truth)]) / len(truth)

def f1_score(precision, recall):
    return 2*(precision*recall)/(precision+recall)

### wsd_classifier() with support for stemming and statistical performance metrics.
def project_classifier(trainer, word, features, stopwords_list = STOPWORDS_SET, number=300, log=False, distance=3, confusion_matrix=False, metrics = False):
    print("Reading data...")
    global _inst_cache
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    print(' Senses: ' + ' '.join(senses))

    # Split the instances into a training and test set,
    #if n > len(events): n = len(events)
    n = len(events)
    random.seed(5444522)
    random.shuffle(events)
    training_data = events[:int(0.8 * n)]
    test_data = events[int(0.8 * n):n]
    # Train classifier
    print('Training classifier...')
    classifier = trainer([(features(i, vocab, distance), label) for (i, label) in training_data])
    # Test classifier
    print('Testing classifier...')
    acc = accuracy(classifier, [(features(i, vocab, distance), label) for (i, label) in test_data] )
    print('Accuracy: %6.4f' % acc)
    if log==True:
        #write error file
        print('Writing errors to errors.txt')
        output_error_file = open('errors.txt', 'w')
        errors = []
        for (i, label) in test_data:
            guess = classifier.classify(features(i, vocab, distance))
            if guess != label:
                con =  i.context
                position = i.position
                item_number = str(test_data.index((i, label)))
                word_list = []
                for (word, tag) in con:
                    word_list.append(word)
                hard_highlighted = word_list[position].upper()
                word_list_highlighted = word_list[0:position] + [hard_highlighted] + word_list[position+1:]
                sentence = ' '.join(word_list_highlighted)
                errors.append([item_number, sentence, guess,label])
        error_number = len(errors)
        output_error_file.write('There are ' + str(error_number) + ' errors!' + '\n' + '----------------------------' +
                                '\n' + '\n')
        for error in errors:
            output_error_file.write(str(errors.index(error)+1) +') ' + 'example number: ' + error[0] + '\n' +
                                    '    sentence: ' + error[1] + '\n' +
                                    '    guess: ' + error[2] + ';  label: ' + error[3] + '\n' + '\n')
        output_error_file.close()
    if confusion_matrix==True:
        gold = [label for (i, label) in test_data]
        derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        cm = nltk.ConfusionMatrix(gold,derived)
        print(cm)
        #return cm
    if metrics:
        gold = [label for (i, label) in test_data]
        derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        for i in senses:
            p = precision(gold, derived, i)
            r = recall(gold, derived, i)
            acc = accuracy_per_label(gold, derived, i)
            f1 = f1_score(p, r)
            print(i, " Precision: ", p, " Recall: ", r, " Accuracy: ", acc, " F1-score: ", f1)
            #print(i, " Precision: %6.4f Recall: %6.4f Accuracy: %6.4f F1-score: %6.4f"%p,r,acc,f1)

if __name__ == "__main__":
    print("NB, with features based on 300 most frequent context words")
    project_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("NB, with features based word + pos in 6 word window")
    project_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_context_features,confusion_matrix=True, metrics=True)
    print("")

    svc = LinearSVC()
    print("SVC, with features based on 300 most frequent context words")
    project_classifier(SklearnClassifier(svc).train, 'hard.pos', wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("SVC, with features based word + pos in 6 word window")
    project_classifier(SklearnClassifier(svc).train, 'hard.pos', wsd_context_features, confusion_matrix=True, metrics=True)
    print("")

    rfc = RandomForestClassifier()
    print("RandomForest, with features based on 300 most frequent context words")
    project_classifier(SklearnClassifier(rfc).train, 'hard.pos', wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("RandomForest, with features based word + pos in 6 word window")
    project_classifier(SklearnClassifier(rfc).train, 'hard.pos', wsd_context_features, confusion_matrix=True, metrics=True)
    print("")

    dtc = DecisionTreeClassifier()
    print("DecisionTree, with features based on 300 most frequent context words")
    project_classifier(SklearnClassifier(dtc).train, 'hard.pos', wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("DecisionTree, with features based word + pos in 6 word window")
    project_classifier(SklearnClassifier(dtc).train, 'hard.pos', wsd_context_features, confusion_matrix=True, metrics=True)
    print("")
