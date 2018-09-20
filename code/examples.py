#CODE TAKEN FROM: http://nlpforhackers.io/named-entity-extraction/


#example of tree to iob format changing

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
 
sentence = "Mark and John are working at Google."
ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
 
iob_tagged = tree2conlltags(ne_tree)
print iob_tagged
"""
[('Mark', 'NNP', u'B-PERSON'), ('and', 'CC', u'O'), ('John', 'NNP', u'B-PERSON'), ('are', 'VBP', u'O'), ('working', 'VBG', u'O'), ('at', 'IN', u'O'), ('Google', 'NNP', u'B-ORGANIZATION'), ('.', '.', u'O')]
"""
 
#############################################################################################
#example of GMB reader - fairly useful but not that hard to do

ne_tree = conlltags2tree(iob_tagged)
print ne_tree

ner_tags = collections.Counter()
 
for root, dirs, files in os.walk(corpus_root):
    for filename in files:
        if filename.endswith(".tags"):
            with open(os.path.join(root, filename), 'rb') as file_handle:
                file_content = file_handle.read().decode('utf-8').strip()
                annotated_sentences = file_content.split('\n\n')   # Split sentences
                for annotated_sentence in annotated_sentences:
                    annotated_tokens = [seq for seq in annotated_sentence.split('\n') if seq]  # Split words
 
                    standard_form_tokens = []
 
                    for idx, annotated_token in enumerate(annotated_tokens):
                        annotations = annotated_token.split('\t')   # Split annotation
                        word, tag, ner = annotations[0], annotations[1], annotations[3]
 
                        # Get only the primary category
                        if ner != 'O':
                            ner = ner.split('-')[0]
 
                        ner_tags[ner] += 1
 
print ner_tags
# Counter({u'O': 1146068, u'geo': 58388, u'org': 48094, u'per': 44254, u'tim': 34789, u'gpe': 20680, u'art': 867, u'eve': 709, u'nat': 300})
 
print "Words=", sum(ner_tags.values())
# Words= 1354149


"""
(S
  (PERSON Mark/NNP)
  and/CC
  (PERSON John/NNP)
  are/VBP
  working/VBG
  at/IN
  (ORGANIZATION Google/NNP)
  ./.)
"""