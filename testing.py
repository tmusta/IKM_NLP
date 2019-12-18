from wsd_code import *
from project import *

if __name__ == "__main__":
  project_classifier(NaiveBayesClassifier.train, 'hard.pos', wsd_word_features, stem=True, confusion_matrix=False, metrics=True)
