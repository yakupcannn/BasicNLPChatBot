import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
import numpy as np

porter_stemmer = nltk.PorterStemmer()

def tokenize(sentence):
  return nltk.word_tokenize(sentence)

def stem(word):
  return porter_stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):

  t_sentences = [stem(_w) for _w in tokenized_sentence if _w not in ["!","?","."]]
  bag = np.zeros(shape = len(all_words),dtype=np.float32)
  
  for i,w in enumerate(all_words):
    if w in t_sentences:
      bag[i] = 1.0
  return bag


