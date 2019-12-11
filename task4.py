from wsd_code import *
from project import *
import os

import plotly.graph_objs as go
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

  

def gen_word_dicts(results):
  #results = [NS300, NS300_NS]
  words = results[list(results.keys())[0]]
  word_dicts = {}
  for i in words:
    word_dicts[i] = {}
    for j in list(results.keys()):
      word_dicts[i][j] = {}
  
  
  for category in list(results.keys()):
    for word in list(results[category].keys()):
      for test in list(results[category][word].keys()):
        word_dicts[word][category][test] = results[category][word][test]

  return word_dicts

    

def no_stops_test(word):
  #word example = "hard.pos"
  
  NB300 = project_classifier(NaiveBayesClassifier.train, word, wsd_word_features, confusion_matrix=True, metrics=True)
  NB300_NS = project_classifier(NaiveBayesClassifier.train, word, wsd_word_features, stopwords_list=NO_STOPWORDS, confusion_matrix=True, metrics=True)

  NBW6 = project_classifier(NaiveBayesClassifier.train, word, wsd_context_features, stopwords_list=STOPWORDS_SET, confusion_matrix=True, metrics=True)
  NBW6_NS = project_classifier(NaiveBayesClassifier.train, word, wsd_context_features, stopwords_list=NO_STOPWORDS, confusion_matrix=True, metrics=True)

  svc = LinearSVC()
  SVC300 = project_classifier(SklearnClassifier(svc).train, word, wsd_word_features, stopwords_list=STOPWORDS_SET,  confusion_matrix=True, metrics=True)
  SVC300_NS = project_classifier(SklearnClassifier(svc).train, word, wsd_word_features, stopwords_list=NO_STOPWORDS,  confusion_matrix=True, metrics=True)
  
  SVCW6  = project_classifier(SklearnClassifier(svc).train, word, wsd_context_features, stopwords_list=STOPWORDS_SET,  confusion_matrix=True, metrics=True)
  SVCW6_NS = project_classifier(SklearnClassifier(svc).train, word, wsd_context_features, stopwords_list=NO_STOPWORDS,  confusion_matrix=True, metrics=True)


  rfc = RandomForestClassifier()
  RFC300 = project_classifier(SklearnClassifier(rfc).train, word, wsd_word_features, stopwords_list=STOPWORDS_SET,  confusion_matrix=True, metrics=True)
  RFC300_NS = project_classifier(SklearnClassifier(rfc).train, word, wsd_word_features, stopwords_list=NO_STOPWORDS,  confusion_matrix=True, metrics=True)
  
  RFCW6 = project_classifier(SklearnClassifier(rfc).train, word, wsd_context_features, stopwords_list=STOPWORDS_SET,  confusion_matrix=True, metrics=True)
  RFCW6_NS = project_classifier(SklearnClassifier(rfc).train, word, wsd_context_features, stopwords_list=NO_STOPWORDS,  confusion_matrix=True, metrics=True)


  dtc = DecisionTreeClassifier()
  DTC300 = project_classifier(SklearnClassifier(dtc).train, word, wsd_word_features, stopwords_list=STOPWORDS_SET,  confusion_matrix=True, metrics=True)
  DTC300_NS = project_classifier(SklearnClassifier(dtc).train, word, wsd_word_features, stopwords_list=NO_STOPWORDS,  confusion_matrix=True, metrics=True)
  
  DTCW6 = project_classifier(SklearnClassifier(dtc).train, word, wsd_context_features, stopwords_list=STOPWORDS_SET,  confusion_matrix=True, metrics=True)
  DTCW6_NS = project_classifier(SklearnClassifier(dtc).train, word, wsd_context_features, stopwords_list=NO_STOPWORDS,  confusion_matrix=True, metrics=True)


def gen_average_table_data(results, category_title, modifiers):
  # results = {CATEGORY1: {WORD1: {acc: 0}}}
  comb_values_base = {}
  header = [category_title, "Pre","d_Pre","Rec","d_Rec","Acc","d_Acc","F1","d_F1"]
  rows = [header]
  for category in list(results.keys()):
    combined_values = {"precision":[],"recall": [],"accuracy": [],"f1": []}

    for word in list(results[category].keys()):
      for test in list(results[category][word].keys()):
        combined_values[test].append(results[category][word][test])
        row = [category]

      if [category.endswith(i) for i in modifiers].count(True) == 0:
        comb_values_base = combined_values

      for test in list(results[category][word].keys()):
        
        #add the rounded average of different senses scores
        average = sum(combined_values[test])/len(combined_values[test])
        row.append(str(round(average,3)))
        #add the delta value
        
        if list(results[category][word].keys()) and [category.endswith(i) for i in modifiers].count(True):
          base_average = sum(comb_values_base[test])/len(comb_values_base[test])
          string = str(round((((average / base_average) * 100) - 100),1))+"%"
          if not string.startswith("-"):
            string = "+"+string
          row.append(string)
        else:
          row.append("")
    print(comb_values_base == combined_values)
    print(row)
    rows.append(row)
  return rows




    



def gen_individual_table_data(results, category_title, modifiers):
  # results = {WORD1: {CATEGORY1: {acc: 0}}}
  indices = [1,2,3,4]
  datas = []
  for word in list(results.keys()):
    header = [category_title+ " " + word + "<br>Precision<br>d_Precision<br>Recall<br>d_Recall<br>Accuracy<br>d_Accuracy<br>F1<br>d_F1"]
    header = [word, "Pre","d_Pre","Rec","d_Rec","Acc","d_Acc","F1","d_F1"]
    rows = [header]
    columns = list(results.keys())
    for category in list(results[word].keys()):
      row = [category]
      
      for test in list(results[word][category].keys()):
        #
        row.append(round(results[word][category][test], 3))
        #this nightmare loop calculates the delta values for the table. modifiers should be a list of all ending modifiers: e.g. ["_NS", "_STEM"]
        
        if list(results[word][category].keys()) and [category.endswith(i) for i in modifiers].count(True):
          row.append(str(round((results[word][category][test] / results[word][category_title][test] * 100) - 100, 1))+"%")
        else:
          row.append("")
        
        
        
      rows.append(row)

    datas.append(rows)
    

  return datas

      


def gen_graph(data, name):
  #results = [[columnd id, column0, column1, ...]]
  ["Test Category","Precision","d_Precision", "Recall", "d_Recall", "Accuracy", "d_Accuracy", "F1", "d_F1"]
  try:
    os.mkdir(os.getcwd()+"/images/")
  except:
    pass
  table = ff.create_table(data)
  plotly.io.write_image(table, os.getcwd()+"/images/"+name)


def gen_NB300_graphs(word):
  #NB, with features based on 300 most frequent context words
  NB300 = project_classifier(NaiveBayesClassifier.train, 
    word, wsd_word_features, 
    confusion_matrix=False,
    metrics=True)

  #Without stopword filtering
  NB300_NS = project_classifier(NaiveBayesClassifier.train, 
    word, wsd_word_features, 
    stopwords_list=NO_STOPWORDS, 
    confusion_matrix=False, 
    metrics=True)

  modifiers = ["_NS"] #no stops modifier. Later no stemming and others.

  results = {"NB300": NB300, "NB300_NS": NB300_NS}
  inverted_results = gen_word_dicts(results)

  avg_data = gen_average_table_data(results, word, modifiers)
  sense_datas = gen_individual_table_data(inverted_results, "NB300", modifiers)

  #Create average graph.
  gen_graph(avg_data, word+"_avg.png")

  #Create graphs for individual senses.
  for i in range(len(sense_datas)):
    gen_graph(sense_datas[i], word+"_"+str(i)+".png")

if __name__ == "__main__":
  word = "hard.pos"
  gen_NB300_graphs(word)