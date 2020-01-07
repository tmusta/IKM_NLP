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
from preprocessing import *
import math

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
    try:
        return sum([x == label and y == label for x, y in zip(preds, truth)]) / sum([x == label for x in truth])
    except ZeroDivisionError:
        return 0.0

def precision(truth, preds, label):
    try:
        return sum([x == label and y == label for x, y in zip(preds, truth)]) / sum([x == label for x in preds])
    
    except ZeroDivisionError:
        return 0.0


def accuracy_per_label(truth, preds, label):
    return sum([(x == label and y == label) or (x != label and y != label) for x, y in zip(preds, truth)]) / len(truth)

def f1_score(precision, recall):
    try:
        return 2*(precision*recall)/(precision+recall)
    
    except ZeroDivisionError:
        return 0.0


def wsd_tfidf_features(instance, vocab, dist=3):
    """
    Create a featureset where every key returns False unless it occurs in the
    instance's context
    """
    words = [w for (w, f) in vocab]
    freqs = [f for (w, f) in vocab]
    features = defaultdict(lambda:False)
    #features['alwayson'] = True
    #cur_words = [w for (w, pos) in i.context]
    try:
        for(w, pos) in instance.context:
            if w in words:
                if not w in features:
                    features[w] = 1
                else:
                    features[w] += 1
        #for w in features:
        #    features[w] /= len(instance.context)
        for w, f in zip(words, freqs):
            if w in features:
                features[w] = math.log(1 + features[w]) * math.log(f)
    except ValueError:
        pass
    return features

def N_gram(sentence, N=2, stopwords=STOPWORDS):
    grams = []
    #sentence = sentence.split(" ")
    sentence = [i for i in sentence if not i in stopwords]
    if len(sentence) < N:
        return grams
    for i in range(len(sentence)- N):
        tmp = ""
        for j in range(N):
            if type(sentence[i + j]) == str:
                tmp += sentence[i + j]
            elif type(sentence[i + j]) == tuple:
                tmp += sentence[i + j][0]
        grams.append(tmp)
    return grams

def extract_vocab_grams_frequency(instances, stopwords=STOPWORDS_SET, n=300):
    fd = nltk.FreqDist()
    for i in instances:
        (target, suffix) = i.word.split('-')
        
        words = (c[0] for c in i.context if not c[0] == target)
        grams = N_gram(words, stopwords=STOPWORDS_SET)
        for g in grams:
            fd[g] += 1
    return fd.most_common()[:n+1]

def extract_vocab_grams(instances, stopwords=STOPWORDS_SET, n=300):
    return [w for w,f in extract_vocab_grams_frequency(instances,stopwords,n)]


def gram_context_features(instance, vocab, dist=3):
    features = {}
    ind = instance.position
    con = instance.context
    for i in range(max(0, ind-dist), ind):
        j = ind-i
        features['left-context-word-%s(%s)' % (j, con[i][0])] = True

    for i in range(ind+1, min(ind+dist+1, len(con))):
        j = i-ind
        features['right-context-word-%s(%s)' % (j, con[i][0])] = True

 
    features['word'] = instance.word
    features['pos'] = con[1][1]
    return features

def gram_word_features(instance, vocab, dist=3):
    """
    Create a featureset where every key returns False unless it occurs in the
    instance's context
    """
    features = defaultdict(lambda:False)
    features['alwayson'] = True
    #cur_words = [w for (w, pos) in i.context]
    try:
        for(w, pos) in instance:
            if w in vocab:
                features[w] = True
    except ValueError:
        pass
    return features



### wsd_classifier() with support for stemming and statistical performance metrics.

def project_classifier(trainer, word, features, stopwords_list=STOPWORDS_SET, stem=False, stopwords_own=False ,ext_words=False, replace_chars=False, remove_empties=False, number=300, no_global_cache=False, log=False, distance=3, confusion_matrix=False, metrics = False, vocab_f=extract_vocab, ngrams=False):


    #NOTICE UPDATED PARAMS, works without changing them also, but they are there.
    print("Reading data...")

    #global_cache screws up calls with different preprocessing params.
    #could probably optimize this so that the data doesn't need to be read every time, but this is used only for the graph generation so doesn't really matter.
    if no_global_cache:
      cache = {}
      cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
      events = cache[word][:]
    else:
      global _inst_cache
      if word not in _inst_cache:
          _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
      events = _inst_cache[word][:]

    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    
    
    #PREPROCESSING ADDED HERE
    if stem or ext_words or replace_chars or remove_empties or stopwords_own:
      stemmer = None
      if stem:
        stemmer = PorterStemmer()
      instances = preprocess_instances(instances, stemmer, stopwords_own, ext_words, replace_chars, remove_empties)

    
    #vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    vocab = vocab_f(instances, stopwords=stopwords_list, n=number)
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
    classifier = None
    acc = 0.0
    if ngrams:
        classifier = trainer([(features(N_gram(i.context), vocab, distance), label) for (i, label) in training_data])
    else:
        classifier = trainer([(features(i, vocab, distance), label) for (i, label) in training_data])
    # Test classifier
    print('Testing classifier...')
    if ngrams:
        acc = accuracy(classifier, [(features(N_gram(i.context), vocab, distance), label) for (i, label) in test_data] )
    else:
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
        derived = None
        if ngrams:
            derived = [classifier.classify(features(N_gram(i.context),vocab)) for (i,label) in test_data]
        else:
            derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        cm = nltk.ConfusionMatrix(gold,derived)
        print(cm)
        #return cm
    if metrics:
        gold = [label for (i, label) in test_data]
        derived = None
        if ngrams:
            derived = [classifier.classify(features(N_gram(i.context),vocab)) for (i,label) in test_data]
        else:
            derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        #derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
        results = {}

        w_acc = acc

        mean_p = 0.0
        mean_r = 0.0
        mean_f1 = 0.0

        for i in senses:
            p = precision(gold, derived, i)
            r = recall(gold, derived, i)
            acc = accuracy_per_label(gold, derived, i)
            f1 = f1_score(p, r)
            mean_p += p
            mean_r += r
            mean_f1 += f1
            print(i, " Precision: ", p, " Recall: ", r, " Accuracy: ", acc, " F1-score: ", f1)
            #print(i, " Precision: %6.4f Recall: %6.4f Accuracy: %6.4f F1-score: %6.4f"%p,r,acc,f1)
            
            #added results dict that returns from the function

            results[i] = {"precision":p,"recall": r,"accuracy": acc,"f1": f1, "w_acc": w_acc}

        print("MEAN_PRECISION: ", mean_p / len(senses), "MEAN_RECALL: ", mean_r / len(senses), "MEAN_F1: ", mean_f1 / len(senses))

        return results

if __name__ == "__main__":
    word = 'serve.pos'
    print("NB, with features based on 300 most frequent context words")
    project_classifier(NaiveBayesClassifier.train, word, gram_word_features, confusion_matrix=True, metrics=True, vocab_f=extract_vocab_grams, ngrams=True)
    project_classifier(NaiveBayesClassifier.train, word, wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    exit()
    print("NB, with features based word + pos in 6 word window")
    project_classifier(NaiveBayesClassifier.train, word, wsd_context_features,confusion_matrix=True, metrics=True)
    print("")
    print("NB, with features based tf_idf")
    project_classifier(NaiveBayesClassifier.train, word, wsd_tfidf_features,confusion_matrix=True, metrics=True, vocab_f=extract_vocab_frequency)
    print("")
    
    svc = LinearSVC()
    print("SVC, with features based on 300 most frequent context words")
    project_classifier(SklearnClassifier(svc).train, word, wsd_word_features, confusion_matrix=True, metrics=True, vocab_f=extract_vocab_grams, ngrams=True)
    print("")
    project_classifier(SklearnClassifier(svc).train, word, wsd_word_features, confusion_matrix=True, metrics=True)
    print("SVC, with features based word + pos in 6 word window")
    project_classifier(SklearnClassifier(svc).train, word, wsd_context_features, confusion_matrix=True, metrics=True)
    print("")
    print("SVC, with features based on tfidf")
    project_classifier(SklearnClassifier(svc).train, word, wsd_tfidf_features, confusion_matrix=True, metrics=True, vocab_f=extract_vocab_frequency)
    """
    rfc = RandomForestClassifier()
    print("RandomForest, with features based on 300 most frequent context words")
    project_classifier(SklearnClassifier(rfc).train, word, wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("RandomForest, with features based word + pos in 6 word window")
    project_classifier(SklearnClassifier(rfc).train, word, wsd_context_features, confusion_matrix=True, metrics=True)
    print("")

    dtc = DecisionTreeClassifier()
    print("DecisionTree, with features based on 300 most frequent context words")
    project_classifier(SklearnClassifier(dtc).train, word, wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("DecisionTree, with features based word + pos in 6 word window")
    project_classifier(SklearnClassifier(dtc).train, word, wsd_context_features, confusion_matrix=True, metrics=True)
    print("")
    """
