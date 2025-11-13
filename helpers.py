import nltk
import csv

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