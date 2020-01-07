from nltk.corpus import senseval, stopwords
from nltk.stem import PorterStemmer
import copy

def get_instances(word):
  cache = {}
  if word not in cache:
    cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
  events = cache[word][:]
  senses = list(set(l for (i, l) in events))
  instances = [i for (i, l) in events]
  return instances

REMOVE_WORDS = ["","''", '""', "``", "¨¨", ".", ",","!","?"]
EXTEND_WORDS = [("'m", "am"), ("'t", "not"), ("'s","is"), ("'re","are"), ("'ve","have"), ("'d","had"), ("",""), ("",""),]
REMOVE_TAGS = ["SYM"]
REPLACE_CHARS = [("à", "a"),("á", "a"),("â","a"),("ã","a"),("ä","a"),("ç","c"),("è", "e"),("é", "e"),("ê","e"),("ë", "e"),("ì", "i"),("í","i"),("î","i"),("ï","i"),("ñ","n"),("ò","o"),("ó","o"),("ô","o"),("õ","o"),("ö","o"),("š","s"),("ù","u"),("ú","u"),("û","u"),("ü","u"),("ý","y"),("ÿ","y"),("ž","z")
, (".",""), (",",""), ("?",""), ("!",""), ("'",""), ('"',""), ("¨",""), ("*",""), ("^",""), ("-",""), (":",""), (";",""), ("(",""), (")",""), ("[",""), ("]",""), ("£",""), ("$",""), ("€","")]
SYMBOLS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(', 
             ')', '$']

STOP_WORDS = stopwords.words("english")


def stopword_removal(instance, stop_word_list):
  to_remove = []
  shift_position = 0
  for index, value in enumerate(instance.context):
    if stop_word_list.count(value[0]):
      to_remove.append(value)
      if index < instance.position:
        shift_position +=1

  for i in to_remove:
    instance.context.remove(i)
  
  instance.position -= shift_position


  return instance

def extend_words(instance, extend_list):
  """
  if context entry equals EXTEND_WORDS[i][0] replace it with EXTEND_WORDS[i][1]
  'm -> am etc.
  """

  for index, value in enumerate(instance.context):
    for original,replacement in extend_list:
      try:
        if value[0] == original:
          instance.context[index] = (replacement, value[1])
      except:
        pass
  return instance

def replace_characters(instance, replace_list):
  """
  replaces characters with substitutes, e.g.  ä => a etc.
  """
  for index, value in enumerate(instance.context):
    for original,replacement in replace_list:
      try:
        instance.context[index] = (instance.context[index][0].replace(original,replacement), value[1])
      except:
        pass
  return instance

def remove_non_words(instance, remove_words, remove_tags=None):
  """
  removes context entries that match remove words or tags,
  calculates the new position for the word.
  """
  #TODO
  to_remove = []
  shift_position = 0
  for index, value in enumerate(instance.context):
    if remove_words.count(value[0]) or remove_tags.count(value[1]):
      to_remove.append(value)
      if index < instance.position:
        shift_position +=1

  for i in to_remove:
    instance.context.remove(i)
  instance.position -= shift_position


  return instance


def stem_instance(instance, stemmer):
  """
  Goes through the instances context and stems the entries.
  """
  for index_context, value in enumerate(instance.context):
    try:
      instance.context[index_context] = (stemmer.stem(value[0]),value[1])
    except:
      pass
  return instance





def preprocess_instances(instances, stemmer=None, stopwords=False, ext_words=False, replace_chars=False, remove_invalid_words=False):
  """
  Performs selected preprocessing operations on all instances
  """

  for index, inst in enumerate(instances):
    if stopwords:
      inst = stopword_removal(inst,STOP_WORDS)
    if ext_words:
      inst = extend_words(inst, EXTEND_WORDS)
    if replace_chars:
      inst = replace_characters(inst, REPLACE_CHARS)
    if remove_invalid_words:
      inst = remove_non_words(inst, REMOVE_WORDS, REMOVE_TAGS)
    if stemmer != None:
      inst = stem_instance(inst, stemmer)
    

  return instances

def print_sentance(senseval_context):
  string = ""
  for i in senseval_context:
    return " ".join(i[0] for i in senseval_context)
  

def demo(instance):
  
  print("\nNo preprocessing:",instance)
  #print("\nStopwords",preprocess_instances([copy.deepcopy(instance)],stopwords=True)[0])
  print("\nStemmed:",preprocess_instances([copy.deepcopy(instance)],PorterStemmer())[0])
  print("\nContraction Reversal:",preprocess_instances([copy.deepcopy(instance)],ext_words=True)[0])
  print("\nReplacing Characters:",preprocess_instances([copy.deepcopy(instance)],replace_chars=True)[0])
  print("\nRemove non_words:",preprocess_instances([copy.deepcopy(instance)],remove_invalid_words=True)[0])
  print("\nAll of the above:",preprocess_instances([copy.deepcopy(instance)],PorterStemmer(), ext_words=True, replace_chars=True,remove_invalid_words=True)[0])
  """
  all_pp = preprocess_instances([copy.deepcopy(instance)],PorterStemmer(), stopwords=True, ext_words=True, replace_chars=True,remove_invalid_words=True)[0]
  actual = all_pp.position
  for i, v in enumerate(all_pp.context):
    if v[0].count("hard"):
      actual = i
      break
  if i != all_pp.position:
    print("pos:",all_pp.position, "actual:", i)
  """

  


if __name__ == "__main__":
  word = "hard.pos"
  instances = get_instances(word)
  
  
  demo(instances[0])

  
  #instances_new = preprocess_instances(instances_new, PorterStemmer(), True, True)


  
  """
  instance = instances[1]
  stemmer = PorterStemmer()
  print(instance)
  instance.context = [(stemmer.stem(word),tag) for (word, tag) in instance.context]
  print(instance)
  """
  #stemmed =  stem_instances(instances)