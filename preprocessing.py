from nltk.corpus import senseval
from nltk.stem import PorterStemmer

def get_instances(word):
  cache = {}
  if word not in cache:
    cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
  events = cache[word][:]
  senses = list(set(l for (i, l) in events))
  instances = [i for (i, l) in events]
  return instances

REMOVE_WORDS = ["","''", '""', "``", "¨¨", ".", ",","!","?"]
REMOVE_TAGS = ["SYM"]
REPLACE_CHARS = [("à", "a"),("á", "a"),("â","a"),("ã","a"),("ä","a"),("ç","c"),("è", "e"),("é", "e"),("ê","e"),("ë", "e"),("ì", "i"),("í","i"),("î","i"),("ï","i"),("ñ","n"),("ò","o"),("ó","o"),("ô","o"),("õ","o"),("ö","o"),("š","s"),("ù","u"),("ú","u"),("û","u"),("ü","u"),("ý","y"),("ÿ","y"),("ž","z")
, (".",""), (",",""), ("?",""), ("!",""), ("'",""), ('"',""), ("¨",""), ("*",""), ("^",""), ("-",""), (":",""), (";",""), ("(",""), (")",""), ("[",""), ("]",""), ("£",""), ("$",""), ("€","")]
SYMBOLS = ['.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$']


def replace_characters(instance, replace_list):
  """
  replaces characters with substitutes, e.g.  ä => a etc.
  """
  #TODO
  for index, value in enumerate(instance.context):
    for original,replacement in replace_list:
      try:
        instance.context[index] = (value[0].replace(original,replacement), value[1])
      except:
        pass
  return instance

def remove_non_words(instance, remove_words, remove_tags=None):
  """
  removes context entries that match remove words or tags,
  calculates the new position for the word.
  """
  #TODO

  for index, value in enumerate(instance.context):
    if remove_words.count(value[0]) or remove_tags.count(value[1]):
      instance.context.remove(value)
      if index + 1 < instance.position:
        instance.position -= 1


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





def preprocess_instances(instances, stemmer=None, replace_chars=False, remove_invalid_words=False):
  """
  Performs selected preprocessing operations on all instances
  """

  for index, inst in enumerate(instances):
    if replace_chars:
      inst = replace_characters(inst, REPLACE_CHARS)
    if remove_invalid_words:
      inst = remove_non_words(inst, REMOVE_WORDS, REMOVE_TAGS)
    if stemmer != None:
      inst = stem_instance(inst, stemmer)

  return instances


if __name__ == "__main__":
  word = "hard.pos"
  instances = get_instances(word)
  instances_new = get_instances(word)
  instances_new = preprocess_instances(instances_new, PorterStemmer(), True, True)
  for i in range(0,3):
    print(print("\n",instances[i]))
    print("\n",instances_new[i],"\n")

  
  """
  instance = instances[1]
  stemmer = PorterStemmer()
  print(instance)
  instance.context = [(stemmer.stem(word),tag) for (word, tag) in instance.context]
  print(instance)
  """
  #stemmed =  stem_instances(instances)