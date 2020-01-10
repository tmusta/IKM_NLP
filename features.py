from wsd_code import *
from project import *
from lesk import *
from task4 import gen_word_dicts, gen_average_table_data_without_delta, gen_graph
import os

import plotly.graph_objects as go
import plotly.figure_factory as ff

import plotly

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

SYMBOLS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$']


def gen_feature_graphs(word):

  results = {}
  svc = LinearSVC()
  c100 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=100,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  
  c200 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=200,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  
  c250 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=250,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  c300 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=300,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  
  c350 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=350,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  c400 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=400,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  c500 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_word_features,
                              number=500,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  w2 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_context_features,
                              distance=1,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  w4 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_context_features,
                              distance=2,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  w6 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_context_features,
                              distance=3,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  w8 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_context_features,
                              distance=4,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  w10 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_context_features,
                              distance=5,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True)
  t100 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=100,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  
  t200 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=200,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  
  t250 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=250,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  t300 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=300,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  
  t350 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=350,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  t400 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=400,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  t500 = project_classifier(SklearnClassifier(svc).train, 
                              word, wsd_tfidf_features,
                              number=500,
                              stopwords_list=NO_STOPWORDS,
                              stem=True,
                              no_global_cache=True,
                              confusion_matrix=False, 
                              metrics=True, vocab_f=extract_vocab_frequency)
  
  results["F100"] = c100
  results["F200"] = c200
  results["F250"] = c250
  results["F300"] = c300
  results["F350"] = c350
  results["F400"] = c400
  results["F500"] = c500

  results["w2"] = w2
  results["w4"] = w4
  results["w6"] = w6
  results["w8"] = w8
  results["w10"] = w10
  
  results["t100"] = t100
  results["t200"] = t200
  results["t250"] = t250
  results["t300"] = t300
  results["t350"] = t350
  results["t400"] = t400
  results["t500"] = t500
  
  inverted_results = gen_word_dicts(results)

  avg_data = gen_average_table_data_without_delta(results, word)
  #Create average graph.
  gen_graph(avg_data, "features_"+word+"_avg.png")
  """
  #print(inverted_results)
  for sense in inverted_results.keys():
    sense_data = gen_individual_table_data_without_deltas(inverted_results[sense], sense)
    gen_graph(sense_data, "lesk_"+sense+".png")
  """



if __name__ == "__main__":

  word = "serve.pos"

  gen_feature_graphs(word)
