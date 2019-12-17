from nltk.wsd import lesk
from nltk.corpus import senseval, wordnet
import re
import numpy as np

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

def extend_list(words):
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
            for a in i.lemmas()[0].antonyms(): #antonyms are accessed only from lemmas. The first lemma is directly related to the word. We ignore the antonyms of synonyms
                
                if a.name() not in extended_words:
                    extended_words.append(a.name())
    return extended_words

def extended_lesk(word, context, sense_limit=None):
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
    extended_context_words = extend_list(orig_context_words)
    ###WE CALCULATE COMMON WORDS WITH SENSES
    for i in target_senses:
        score = sum([int(x in i) for x in extended_context_words])
        target_sense_scores.append(score)

    #dummy, argmax = max(target_sense_scores)
    argmax = np.argmax(target_sense_scores)
    return target_synsets[argmax]

if __name__=="__main__":
    word = 'hard.pos'

    if word not in _inst_cache:
        _inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
    events = _inst_cache[word][:]
    senses = list(set(l for (i, l) in events))
    instances = [i for (i, l) in events]
    #vocab = extract_vocab(instances, stopwords=stopwords_list, n=number)
    asd = [i for (i, j) in instances[0].context]
    #print(lesk(asd, 'hard').lemmas()[0].antonyms())
    #print("")
    #for n,i in enumerate(wordnet.synsets('hard')):
    #    print(n, i, i.definition())
    query = "hard"
    context = "the rock is so hard that it broke my hand when I hit it"
    print(query,",",context)
    print("standard lesk: ", lesk(context.split(" "),query).definition())
    print("extended lesk: ", extended_lesk(query, context).definition())
