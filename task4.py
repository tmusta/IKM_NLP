from wsd_code import *
from project import *
from lesk import *
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

    


def gen_average_table_data(results, category_title, modifiers):
  # results = {CATEGORY1: {WORD1: {acc: 0}}}
  comb_values_base = {}
  header = [category_title, "Precision","ΔPrecision","Recall","ΔRecall","Accuracy","ΔAccuracy","F1","ΔF1"]
  rows = [header]
  for category in list(results.keys()):
    combined_values = {"precision":[],"recall": [],"accuracy": [],"f1": []}

    for word in list(results[category].keys()):
      for test in list(results[category][word].keys()):
        if test != "w_acc":
          combined_values[test].append(results[category][word][test])
          row = [category]

      if [category.endswith(i) for i in modifiers].count(True) == 0:
        comb_values_base = combined_values

      for test in list(results[category][word].keys()):
        
        #add the rounded average of different senses scores
        if test == "accuracy":
          row.append(str(round(results[category][word]["w_acc"],3)))
          if list(results[category][word].keys()) and [category.endswith(i) for i in modifiers].count(True):
            base_average = results[category.split("_")[0]][word]["w_acc"]
            string = str(round((((results[category][word]["w_acc"] / base_average) * 100) - 100),1))+"%"
            if not string.startswith("-"):
              string = "+"+string
            row.append(string)
          else:
            row.append("")
        elif test == "w_acc":
          pass
        else:
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
    print(row)
    rows.append(row)
  return rows


def gen_average_table_data_without_delta(results, category_title):
  header = [category_title, "Precision","Recall","Accuracy","F1"]
  rows = [header]
  print(results)
  for category in list(results.keys()):
    combined_values = {"precision":[],"recall": [],"accuracy": [],"f1": []}

    for word in list(results[category].keys()):

      for test in list(results[category][word].keys()):
        if test != "w_acc":
          combined_values[test].append(results[category][word][test])
          row = [category]

      for test in list(results[category][word].keys()):
        if test == "accuracy":
          row.append(str(round(results[category][word]["w_acc"],3)))
        elif test == "w_acc":
          pass
        else:
          average = sum(combined_values[test])/len(combined_values[test])
          row.append(str(round(average,3)))
    print(row)
    rows.append(row)
  return rows

    



def gen_individual_table_data(results, category_title, modifiers):
  # results = {WORD1: {CATEGORY1: {acc: 0}}}
  indices = [1,2,3,4]
  datas = []
  for word in list(results.keys()):
    #header = [category_title+ " " + word + "<br>Precision<br>d_Precision<br>Recall<br>d_Recall<br>Accuracy<br>d_Accuracy<br>F1<br>d_F1"]
    header = [word, "Precision","ΔPrecision","Recall","ΔRecall","Accuracy","ΔAccuracy","F1","ΔF1"]
    rows = [header]
    columns = list(results.keys())
    for category in list(results[word].keys()):
      row = [category]
      
      for test in list(results[word][category].keys()):
        #
        if test != "w_acc":
          row.append(round(results[word][category][test], 3))
          #this nightmare loop calculates the delta values for the table. modifiers should be a list of all ending modifiers: e.g. ["_NS", "_STEM"]
          
          if list(results[word][category].keys()) and [category.endswith(i) for i in modifiers].count(True):
            try:
              string = str(round((results[word][category][test] / results[word][category_title][test] * 100) - 100, 1))+"%"
            except ZeroDivisionError:
              string = "NaN"
            if not string.startswith("-"):
              string = "+"+string
            row.append(string)
          else:
            row.append("")
          
      rows.append(row)
    datas.append(rows)
    

  return datas

def gen_individual_table_data_without_deltas(results, sense):
  datas = []

  #header = [category_title+ " " + word + "<br>Precision<br>d_Precision<br>Recall<br>d_Recall<br>Accuracy<br>d_Accuracy<br>F1<br>d_F1"]
  header = [sense, "Precision","Recall","Accuracy","F1"]
  rows = [header]
  columns = list(results.keys())
  for category in list(results.keys()):
    row = [category]

    
    for test in list(results[category].keys()):
      #
      if test != "w_acc":
        row.append(round(results[category][test], 3))
        
    rows.append(row)
  return rows

def gen_graph2(data, name):
  try:
    os.mkdir(os.getcwd()+"/images/")
  except:
    pass
  fig = go.Figure(data=[go.Table(header=dict(values=data[0], line_color="gray", fill_color="lightgray", align="left"), cells=dict(values=data[0:]))])
  fig.write_image(os.getcwd()+"/images/"+name)


def gen_graph(data, name):
  #results = [[columnd id, column0, column1, ...]]
  #["Test Category","Precision","d_Precision", "Recall", "d_Recall", "Accuracy", "d_Accuracy", "F1", "d_F1"]
  try:
    os.mkdir(os.getcwd()+"/images/")
  except:
    pass
  table = ff.create_table(data)

  plotly.io.write_image(table, os.getcwd()+"/images/"+name)

def gen_classifier_graphs(word, trainers_and_features):
  results_standard = senseval_lesk(word, standard_lesk, NO_STOPWORDS, metrics=True)
  results_extended = senseval_lesk(word, extended_lesk, NO_STOPWORDS, metrics=True)
  results_extended_examples = senseval_lesk(word, extended_lesk, NO_STOPWORDS, metrics=True, use_examples=True)
  results = {}

  for index, name in enumerate(trainers_and_features):
    result = project_classifier(trainers_and_features[name][0], word, trainers_and_features[name][1], no_global_cache=True, metrics=True)
    results[name] = result
  
  

  results["s_lesk"] = results_standard
  results["e_lesk"] = results_extended
  results["e_lesk_ex"] = results_extended_examples
  inverted_results = gen_word_dicts(results)

  avg_data = gen_average_table_data_without_delta(results, word)
  #Create average graph.
  gen_graph(avg_data, "lesk_"+word+"_avg.png")
  
  #print(inverted_results)
  for sense in inverted_results.keys():
    sense_data = gen_individual_table_data_without_deltas(inverted_results[sense], sense)
    gen_graph(sense_data, "lesk_"+sense+".png")



def gen_graphs(word, trainers_and_features, avg_only=False):
  # word = "hard.pos" or something similar
  #trainers_and_features = {name: (trainer_function, features_function)}

  for index, name in enumerate(trainers_and_features):
    #no preprocessing
    org = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      no_global_cache=True,
      confusion_matrix=False,
      metrics=True)

    #with stopword filtering
    stop_SW = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)
    """
    stop2 = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      stopwords_own=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)
    """

    extend = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      ext_words=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)
    
    #remove empty entries from context sentance e.g. ("''")
    remove_RE = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      remove_empties=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)

    #replace accented characters (replace accents)
    replace_RC = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      replace_chars=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)

    #stem entries (stem)
    stem_ST = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      stem=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)


    #all of the above (preprocessing)
    all_ALL = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      remove_empties=True,
      replace_chars=True,
      stem=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)
    """
    all_ALL2 = project_classifier(trainers_and_features[name][0], 
      word, trainers_and_features[name][1],
      stopwords_list=NO_STOPWORDS,
      remove_empties=True,
      stopwords_own=True,
      replace_chars=True,
      stem=True,
      no_global_cache=True,
      confusion_matrix=False, 
      metrics=True)
    """

    

    modifiers = ["_NS", "_SW", "_RE", "_RC", "_ST", "_CR", "_ALL"]

    results = {name: org, name+"_SW": stop_SW, name+"_RE": remove_RE, name+"_RC": replace_RC, name+"_ST": stem_ST, name+"_CR": extend , name+"_ALL": all_ALL}
    inverted_results = gen_word_dicts(results)

    avg_data = gen_average_table_data(results, word, modifiers)
    #Create average graph.
    gen_graph(avg_data, name+"_"+word+"_avg.png")

    #Create graphs for individual senses.
    if not avg_only:
      sense_datas = gen_individual_table_data(inverted_results, name, modifiers)
      for i in range(len(sense_datas)):
        gen_graph(sense_datas[i], name+"_"+word+"_"+str(i)+".png")



if __name__ == "__main__":
  word = "hard.pos"
  svc = LinearSVC()
  rfc = RandomForestClassifier()
  dtc = DecisionTreeClassifier()
  trainers_and_features = {
    #"NB300": (NaiveBayesClassifier.train, wsd_word_features),
    "NB": (NaiveBayesClassifier.train, wsd_context_features),
    
    #"SVC300": (SklearnClassifier(svc).train, wsd_word_features),
    "SVC" : (SklearnClassifier(svc).train, wsd_context_features),

    #"RFC300": (SklearnClassifier(rfc).train, wsd_word_features),
    "RFC" : (SklearnClassifier(rfc).train, wsd_context_features),

    #"DTC300": (SklearnClassifier(dtc).train, wsd_word_features),
    "DTC" : (SklearnClassifier(dtc).train, wsd_context_features),
    
    }

  #Takes a while.
  #gen_graphs(word, trainers_and_features, avg_only=True)
  gen_classifier_graphs(word, trainers_and_features)
