import nltk
import csv
from sklearn.model_selection import train_test_split

nltk.download('punkt')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   space_char: str = ' ',
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  inner_pieces = None
  if by_char:
    line = line.replace(' ', space_char)
    inner_pieces = list(line)
  else:
    # otherwise use nltk's word tokenizer
    inner_pieces = nltk.word_tokenize(line)

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


def read_file(datapath, ngram, by_character=False):
    '''Reads and Returns the "data" as list of list (as shown above)'''
    data = []
    with open(datapath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([
              tokenize_line(row['clean_text'].lower(), ngram, by_char=by_character, space_char="_"),
              int(row['is_depression'])
            ])
    return data

def split(data, dist="80/10/10"):
  print(f"Completing {dist} split")
  test_size = 0.80
  if dist == "70/15/15":
    test_size = 0.70

  train, temp = train_test_split(data, test_size=(1 - test_size), random_state=42)
  dev, test = train_test_split(temp, test_size=0.50, random_state=42)

  return train, dev, test


def get_stats(data):
  # get analytics about data
  n = len(data)

  tokens = []
  vocab = set()
  for row in data:
      for word in row[0]:
          vocab.add(word)
          tokens.append(word)

  # calculate pos and neg depression classifications
  dep_pos = 0
  dep_neg = 0
  for row in data:
      if row[1] == 1:
          dep_pos += 1
      if row[1] == 0:
          dep_neg += 1

  print("======================================================================")
  print(f"Number of lines: {n}")
  print(f"Number of tokens: {len(tokens)}")
  print(f"Number of unique tokens: {len(vocab)}")
  print(f"Number of YES depression entries: {dep_pos} ({dep_pos/n * 100:2.2f}%)")
  print(f"Number of NOT depression entries: {dep_neg} ({dep_neg/n * 100:2.2f}%)")
  print("======================================================================")
