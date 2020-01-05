import nltk
from nltk.wsd import lesk
from nltk.corpus import senseval, wordnet#, accuracy
from nltk.classify import NaiveBayesClassifier
import re
import numpy as np
from wsd_code import extract_vocab, wsd_word_features, wsd_context_features
from project import precision, recall, accuracy_per_label, f1_score, project_classifier

synset_to_senseval = {"hard.pos":{'difficult.a.01': 'HARD1',
                                  'hard.a.02': 'HARD2',
                                  'hard.a.03': 'HARD3'
},
                      "interest.pos": { "interest.n.01" : "interest_1",
                                        "interest.n.03" : "interest_2",
                                        "pastime.n.01" : "interest_3",
                                        "sake.n.01" : "interest_4",
                                        "interest.n.05" : "interest_5",
                                        "interest.n.04" : "interest_6"
                      },
                      "serve.pos": {
                          "serve.v.02" : "SERVE12", # do duty or hold offices; serve in a specific function
                          "serve.v.06" : "SERVE10", # provide (usually but not necessarily food)
                          "serve.v.01" : "SERVE2",  # serve a purpose, role, or function
                          "service.v.01" : "SERVE6"  # be used by; as of a utility
                      },
                      "line.pos": {
                          "line.n.18" : "cord", # something (as a cord or rope) that is long and thin and flexible
                          "line.n.01" : "formation", # a formation of people or things one beside another
                          "line.n.03" : "formation",
                          "line.n.05" : "text", # text consisting of a row of words written across a page or computer screen
                          "telephone_line.n.02" : "phone", # a telephone connection
                          "line.n.22": "product", # a particular kind of product or merchandise
                          "line.n.29" : "division" # a conceptual separation or distinction
                      }
}
    
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

#def wordnet_to_senseval(word):
def remove_useless_symbols(w):
    return re.sub(r'\W+', '', w)

def extend_list(words, use_examples=False):
    extended_words = []
    for w in words:
        for i in  wordnet.synsets(w):
            for s in i.lemma_names():
                if s not in extended_words:
                    extended_words.append(s)
            for h in i.hyponyms():
                for s in h.lemma_names():
                    if s not in extended_words:
                        extended_words.append(s)
            for h in i.hypernyms():
                for s in h.lemma_names():
                    if s not in extended_words:
                        extended_words.append(s)
            if use_examples:
            	for e in i.examples():
            	    for s in e.split(' '):
            	        if s not in extended_words:
            	            extended_words.append(s)
                
            
            for a in i.lemmas()[0].antonyms(): #antonyms are accessed only from lemmas. The first lemma is directly related to the word. We ignore the antonyms of synonyms
                
                if a.name() not in extended_words:
                    extended_words.append(a.name())
    return extended_words

def get_senseval_synsets(word):
    trgt_synsets = synset_to_senseval[word]
    synsets = []
    for i in wordnet.synsets(word.split('.')[0]):
    	for j in trgt_synsets:
    	    if j == i.name():
    	        synsets.append(i)
    return synsets

def extended_lesk(word, context, sense_limit=None, use_examples=False):
    ###PERFORMS EXTENDED LESK
    ### word = target word
    ### context = paragraph or sentence with the word in context
    
    ### returns <class 'nltk.corpus.reader.wordnet.Synset'> object with the predicted sense

    ### Extended LESK extends the words in the paragraph with their hyponyms, direct hypernyms, antonyms and synonyms.
    
    target_synsets = []
    target_senses = []
    target_sense_scores = []
    for n,i in enumerate(wordnet.synsets(word)):
        target_senses.append(i.definition().split(" "))
        target_synsets.append(i)
    if sense_limit:
        target_senses = target_senses[:sense_limit]
        target_synsets = target_synsets[:sense_limit]
    ### WE EXTEND THE CONTEXT WORDS
    orig_context_words = context.split(" ")
    extended_context_words = extend_list(orig_context_words, use_examples=use_examples)
    ###WE CALCULATE COMMON WORDS WITH SENSES
    for i in target_senses:

        score = sum([int(x in i) for x in extended_context_words])
        target_sense_scores.append(score)
    
    #dummy, argmax = max(target_sense_scores)
    argmax = np.argmax(target_sense_scores)
    return target_synsets[argmax]

def standard_lesk(word, context, synsets=None, use_examples=None):
    
    if synsets:
    	synsets = get_senseval_synsets(word)
    return lesk(context.split(" "), word, synsets)

### wsd_classifier() with support for stemming and statistical performance metrics.
def senseval_lesk(word, f, stopwords_list=STOPWORDS_SET, number=300, confusion_matrix=False, metrics = False, use_examples=False):
    print("Reading data...")
    global _inst_cache
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    print(' Senses: ' + ' '.join(senses))
    n = len(events)
    test_data = events#[0:int(0.2 * n)]

    test_data = [(" ".join([j[0] for j in i[0].context]), i[0].senses[0]) for i in test_data]
    label_translator = synset_to_senseval[word]
    correct = 0.0
    preds = []
    for i, j in test_data:

        pred = ""
        """
        if mode == "standard":
            pred = standard_lesk(word.split(".")[0], i)
        else:
            pred = extended_lesk(word.split(".")[0], i, sense_limit=3)
        """
        pred = f(word.split(".")[0], i, use_examples=use_examples)
        #print(pred.name(), j)
        if pred.name() in label_translator:
            preds.append(label_translator[pred.name()])
            #if label_translator.index(pred.name())+1 == int(j[-1]):
            if label_translator[pred.name()] == j:
                correct += 1
        else:
            preds.append(pred.name())
    acc = correct / len(test_data)
    #acc = accuracy(classifier, [(features(i, vocab, distance), label) for (i, label) in test_data] )
    print('Accuracy: %6.4f' % acc)

    if confusion_matrix==True:
        gold = [label for (i, label) in test_data]
        #print(word, test_data[0][0])
        #print(f(word, test_data[0][0]))
        #derived = [f(word.split(".")[0], i).name() for (i,label) in test_data]
        derived = [i for i in preds]
        cm = nltk.ConfusionMatrix(gold,derived)
        print(cm)
        #return cm
    if metrics:
        gold = [label for (i, label) in test_data]
        print(test_data[0])
        #derived = [f(word.split(".")[0], i).name() for (i,label) in test_data]
        derived = [i for i in preds]
        results = {}
        for i in senses:
            p = precision(gold, derived, i)
            r = recall(gold, derived, i)
            acc = accuracy_per_label(gold, derived, i)
            f1 = f1_score(p, r)
            print(i, " Precision: ", p, " Recall: ", r, " Accuracy: ", acc, " F1-score: ", f1)
            #print(i, " Precision: %6.4f Recall: %6.4f Accuracy: %6.4f F1-score: %6.4f"%p,r,acc,f1)
            
            #added results dict that returns from the function
            results[i] = {"precision":p,"recall": r,"accuracy": acc,"f1": f1}
        return results



if __name__=="__main__":
    word = 'serve.pos'
    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    #vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    asd = [i for (i, j) in instances[0].context]
    #print(lesk(asd, 'hard').lemmas()[0].antonyms())

    """
    for n,i in enumerate(wordnet.synsets('interest')):
        print(n, i, i.definition())
    """
    query = "hard.pos"
    context = "It's hard on my vocal chords"
    

    #print(query,",",context)
    #print("standard lesk: ", standard_lesk(query, context, synsets=True).definition())
    #print("extended lesk: ", extended_lesk(query, context).definition())

    ###TASK 5 & 6
    
    print("Standard Lesk")
    print(senseval_lesk(word, standard_lesk), metrics=True, confusion_matrix=True))
    print("Extended Lesk")
    print(senseval_lesk(word, extended_lesk), metrics=True, confusion_matrix=True))
    print("Extended Lesk with examples")
    print(senseval_lesk(word, extended_lesk, use_examples=True, metrics=True, confusion_matrix=True))
    
    print("NB, with features based on 300 most frequent context words")
    project_classifier(NaiveBayesClassifier.train, word, wsd_word_features, confusion_matrix=True, metrics=True)
    print("")
    print("NB, with features based word + pos in 6 word window")
    project_classifier(NaiveBayesClassifier.train, word, wsd_context_features,confusion_matrix=True, metrics=True)
    print("")
    

