from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from nltk.tree import Tree
import json

def build_synset_sentence_index():
    """Preprocess SemCor to create an index of Synsets to sentences."""
    synset_to_sentences = {}

    for sent_tagged, sent in zip(semcor.tagged_sents(tag='both'), semcor.sents()):

        for chunk in sent_tagged:
            if isinstance(chunk, Tree) and hasattr(chunk.label(), 'synset') and chunk.label().synset():
                synset = chunk.label().synset()
                synset_string = synset.name()
                
                if synset_string not in synset_to_sentences:
                    synset_to_sentences[synset_string] = []
                    sentence_text = ' '.join(w for w in sent)  # Construct the sentence text
                synset_to_sentences[synset_string].append(sentence_text)
    
    with open("synset_to_sentences_index.json", "w") as file:
        json.dump(synset_to_sentences, file)


build_synset_sentence_index()

