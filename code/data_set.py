#!/usr/bin/env python

from collections import Iterable
import os
import os.path
import string

from nltk.chunk import ChunkParserI
from nltk.chunk import conlltags2tree
from nltk.tag import ClassifierBasedTagger
from nltk.stem.snowball import SnowballStemmer


def read_dataset(dataset_file):
    with open(dataset_file, 'r', encoding='utf-8', errors='replace') as file_handle:
        current_sentance_num = '1.0'
        current_sentance = []
        for raw_line in file_handle:

            line_contents = raw_line.strip().split()
            sentance_num, word, pos, tag = line_contents[1:]

            if (word, pos, tag) == ('Word', 'POS', 'Tag'):
                continue
            current_sentance.append((word, pos, tag))
            if sentance_num != current_sentance_num:
                current_sentance_num = sentance_num
                full_sentance = current_sentance
                current_sentance = []
                yield [((w, p), t) for w, p, t in full_sentance]


def to_conll_iob(annotated_sentence):
    """
    `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]
    Transform a pseudo-IOB notation: O, PERSON, PERSON, O, O, LOCATION, O
    to proper IOB notation: O, B-PERSON, I-PERSON, O, O, B-LOCATION, O
    """
    proper_iob_tokens = []
    for idx, annotated_token in enumerate(annotated_sentence):
        pos, word, tag = annotated_token
 
        if tag != 'O':
            if idx == 0:
                tag = "B-" + tag
            elif annotated_sentence[idx - 1][2] == tag:
                tag = "I-" + tag
            else:
                tag = "B-" + tag
        proper_iob_tokens.append((pos, word, tag))
    return proper_iob_tokens
 
 
def read_gmb(corpus_root):
    """This function uses os.walk to look through all GMB data files in the 
    root directory of th data, formateed as ((word, pos), tag)
    """
    for root, dirs, files in os.walk(corpus_root):
        for filename in files:
            if filename.endswith(".tags"):
                with open(os.path.join(root, filename), 'rb') as file_handle:
                    file_content = file_handle.read().decode('utf-8').strip()
                    annotated_sentences = file_content.split('\n\n')
                    for annotated_sentence in annotated_sentences:
                        annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]
 
                        standard_form_tokens = []
 
                        for idx, annotated_token in enumerate(annotated_tokens):
                            annotations = annotated_token.split('\t')
                            #was being parsed incorectly tag is annotations[4]
                            word, pos, tag = annotations[0], annotations[1], annotations[4]
                            #print(tag)
                            if tag != 'O':
                                tag = tag.split('-')[1]
 
                            if pos in ('LQU', 'RQU'):   # Make it NLTK compatible
                                pos = "``"

                            standard_form_tokens.append((pos, word, tag))
 
                        conll_tokens = to_conll_iob(standard_form_tokens)
 
                        # Make it NLTK Classifier compatible - [(w1, t1, iob1), ...] to [((w1, t1), iob1), ...]
                        # Because the classfier expects a tuple as input, first item input, second the class
                        yield [((w, p), t) for w, p, t in conll_tokens]


def features_basic(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, p1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders -- to avoid index out of bounds errors
    tokens = [('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]')]
    history = ['[START1]'] + list(history)
 
    # shift the index to accommodate the padding
    index += 1
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    nextword, nextpos = tokens[index + 1]
    previob = history[index - 1]
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
        'prev-iob': previob,
    }


def features_middle(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, p1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders -- to avoid index out of bounds errors
    tokens = [('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]')]
    history = ['[START1]'] + list(history)
 
    # shift the index to accommodate the padding
    index += 1
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    nextword, nextpos = tokens[index + 1]
    previob = history[index - 1]

    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'pos': pos,
 
        'next-word': nextword,
        'next-pos': nextpos,
 
        'prev-word': prevword,
        'prev-pos': prevpos,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }


def features_complicated(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, p1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders -- to avoid index out of bounds errors
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
    history = ['[START2]', '[START1]'] + list(history)
 
    # shift the index with 2, to accommodate the padding
    index += 2
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    previob = history[index - 1]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
        'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev-iob': previob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
    }

 
class NamedEntityChunker(ChunkParserI):

    """Class with overridden parser and init. This class is equipped to learn and predict given
    training data. The data is [[()]]

    """
    def __init__(self, train_sents, feat_detector, **kwargs):

        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = feat_detector
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=feat_detector,
            **kwargs)
 
    def parse(self, tagged_sent):
        """This function is used by evaluate to make guesses and format the guesses
        """
        #make gueess (tag)
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, p, t) for ((w, p), t) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)


def train_and_score(featureset, training_samples, test_samples, other_samples=None):
    """This function takes in training and test data and a featureset and will return
    the scores for testing and training data
    Optional extra set of samples can be sent to be tested against as well
    """
    chunker = NamedEntityChunker(training_samples, featureset)

    testscore = chunker.evaluate([conlltags2tree([(w, pos, tag) for (w, pos), tag in sentance]) 
                                                    for sentance in test_samples]).accuracy()
    trainscore = chunker.evaluate([conlltags2tree([(w, pos, tag) for (w, pos), tag in sentance]) 
                                                    for sentance in training_samples]).accuracy()
    if other_samples:
        mixscore  = chunker.evaluate([conlltags2tree([(w, pos, tag) for (w, pos), tag in sentance]) 
                                                    for sentance in other_samples]).accuracy()
        return (testscore, trainscore, mixscore)
    else:
        return (testscore, trainscore)


if __name__ == '__main__':

    corpus_root = '/home/eoshea/sflintro/gmbdata/gmb-1.0.0'
    data_file = "/home/eoshea/sflintro/data/dataset_22nov17.txt" 

    reader = read_gmb(corpus_root)
    gmbdata = list(reader)

    filedata = list(read_dataset(data_file))

    scores = {}
    feature_types = [('Basic', features_basic), 
                     ('Middle',features_middle), 
                     ('Complicated',features_complicated)]

    #k fold cross-validation
    k_fold = 10
    data_volume = 100    #this is the number of data point per fold: ideally len(data) // k
 
    for feat_key, featureset in feature_types:

        scores[feat_key] = {'GMB': [], 'FILE': [], 'MIX': []}

        for i in range(k_fold):
            ################################### GMB DATA ###################################
            training_samples = gmbdata[0:i*data_volume] + gmbdata[(i+1)*data_volume:k_fold*data_volume]
            test_samples = gmbdata[i*data_volume:(i+1)*data_volume]
            other_samples = filedata[:k_fold*data_volume] 

            scores[feat_key]['GMB'].append(train_and_score(featureset, training_samples, 
                                                           test_samples, other_samples))

            ################################### DATASET DATA ###################################
            training_samples = filedata[0:i*data_volume] + filedata[(i+1)*data_volume:k_fold*data_volume]
            test_samples = filedata[i*data_volume:(i+1)*data_volume]
            other_samples = filedata[:k_fold*data_volume]

            scores[feat_key]['FILE'].append(train_and_score(featureset, training_samples, 
                                                           test_samples, other_samples))

            ################################### MIXED DATA ###################################
            training_samples = gmbdata[0:i*data_volume] + gmbdata[(i+1)*data_volume:k_fold*data_volume]\
                             + filedata[0:i*data_volume] + filedata[(i+1)*data_volume:k_fold*data_volume]

            test_samples = gmbdata[i*data_volume:(i+1)*data_volume] \
                         + filedata[i*data_volume:(i+1)*data_volume]

            scores[feat_key]['MIX'].append(train_and_score(featureset, training_samples, 
                                                           test_samples))


    with open("/home/eoshea/sflintro/scores/scores.json", "w") as outfile:
        json.dump(scores, outfile, indent=4)