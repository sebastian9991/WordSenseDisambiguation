from loader import load_instances, load_key
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
from nltk import Tree
from nltk.wsd import lesk
from nltk.corpus import stopwords
from tabulate import tabulate

'''
Convert the wordnet keys to Synets. We assume this is the correct label
FORM: group%1:03:00:: 
'''
def key_to_synets(word_key):
    return wn.synset_from_sense_key(word_key)
'''
Sense indicates as #1 in the synset according to WordNet
'''
def most_frequent_baseline(lemma, pos = None): 
    synsets = wn.synsets(lemma, pos = pos)
    #Return the first synset
    return synsets[0] if synsets else None

'''
NLTK implementation of Lesk's WSD
'''
def lesk_wsd(context, lemma, pos = None):
    return lesk(context_sentence=context, ambiguous_word= lemma
                , pos=pos)

'''
Method A.
Starter code from: https://www.nltk.org/_modules/nltk/wsd.html#lesk
This method refers from the Corpus Lesk algorithm in the textbook; Section 20.4 (Jurafsky, D. 2nd)
We add two things to the original Lesk algorithm: 
1. Increase the signature set size by including more sentences of each sense. (e.g. from SemCor)
2. Use Inverse Document Frequency to weight each overlapping word
'''
def corpus_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, lang="eng"):
    """Return a synset for an ambiguous word in a context.
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wn.synsets(ambiguous_word, lang=lang)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None
    
    ##NOTE: 
    ## then I would need to use inverse document frequency or IDF
    _, sense = max(
        (len(context.intersection((ss.definition() + 
            ' '.join(s for s in get_semcor_sentences_for_synset(ss))).split())), ss) for ss in synsets
    )
    return sense
    
'''
Checks the equality of two Synsets. 
We employ the use of wn.path_similarity() returns [0, 1]
If they are equal we have 1, otherwise ranges between [0, 1)
'''
def score_synset_lists(syn_1_l, syn_2_l):
    sum = 0
    for syn_1, syn_2 in zip(syn_1_l, syn_2_l):
        sum += syn_1.path_similarity(syn_2)

    return sum / len(syn_1_l)



'''
Download some nltk tools and get stopwords
'''
def get_nltk_tools():
    #Create tools
    stop_words = stopwords.words('english')
    return stop_words

'''
Preprocess the context sentence, namely remove the stop words
'''
def preprocess_context_sentence(context_list):
    stop_words = get_nltk_tools()
    stop_words = set(stop_words)
    
    context_list_preprocessed = []
    for w in context_list:
        if w not in stop_words:
            context_list_preprocessed.append(w)
    
    return context_list_preprocessed

'''
Maps the Penn Treebank Tags to given wn tags
'''
def pos_tag_mapper(tag):
    if tag in {"NN", "NNS", "NNP", "NNPS"}:
        return wn.NOUN
    elif tag in {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}:
        return wn.VERB
    elif tag in {"JJ", "JJR", "JJS"}:
        return wn.ADJ
    elif tag in {"RB", "RBR", "RBS"}:
        return wn.ADV
    return None

'''
We use SemCor a subset of the Brown corpus tagged senses to produce corpus
added to the signature of each sense. 
'''
def get_semcor_sentences_for_synset(synset): 
    sentences = []
    
    for sent_tagged, sent in zip(semcor.tagged_sents(tag='both'), semcor.sents()): 
        for chunk in sent_tagged:
            if isinstance(chunk, Tree): 
                if hasattr(chunk.label(), 'synset'):
                    chunk_synset = chunk.label().synset()
                    if chunk_synset.path_similarity(synset) == 1:
                        #Then they are equal
                        single_string = ' '.join(w for w in sent)
                        sentences.append(single_string)
                        break
        if len(sentences) >= 5: 
            #We limit the extra corpus to 5 sentences for efficency.
            break
    
    return sentences


'''
Yarowasky's bootstrap method refered from 
Speech & Language Processing, Jurafsky, D.2nd (Page 650)
'''
def yarowsky_bootstrap(X):
    ##Define a heuristic for automatic labelling
    seedset = 0




    return 0
    

'''
used from @jcheung start code
Main method
'''
if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('semcor')
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

    ##NOTE: 
    ##We could use online learning by allowing it to generate similar sentence, for one word sense
    ## Then use Yarowsky on the generated X. 

    display_dict = {
        "most_frequent_baseline": 0,
        "lesk_score_raw": 0,
        "lesk_score_pos" : 0,
        "lesk_score_preprocessed": 0, 
        "lesk_score_preprocessed_pos": 0
    }

    y_correct_labels = []
    x_most_frequent_baseline = []
    x_lesk_syn_predictions = []
    x_lesk_syn_predictions_pos = []
    x_lesk_syn_predictions_pre = []
    x_lesk_syn_predictions_pre_pos = []
    for item, key in zip(dev_instances.items(), dev_key):
        syn_key = dev_key[key][0]
        y_correct_labels.append(key_to_synets(syn_key))

        ##Lesk() requires strings not bytes, throws error when input is b''
        wsd_instance = item[1]
        context = list(map(lambda x: x.decode("utf-8"), wsd_instance.context))
        lemma = wsd_instance.lemma.decode("utf-8")
        pos = wsd_instance.pos.decode("utf-8")
        x_most_frequent_baseline.append(most_frequent_baseline(lemma))
        x_lesk_syn_predictions.append(lesk_wsd(context, lemma))
        x_lesk_syn_predictions_pos.append(lesk_wsd(context, lemma, pos = pos_tag_mapper(pos)))
        
        ##After preprocessing
        context_pre = preprocess_context_sentence(context_list=context)
        x_lesk_syn_predictions_pre.append(lesk_wsd(context_pre, lemma))
        x_lesk_syn_predictions_pre_pos.append(lesk_wsd(context=context_pre, lemma = lemma, pos=pos_tag_mapper(pos)))

        #My Methods:
        #METHOD A:
        ##corpus_lesk(context, lemma)

        




    
    display_dict['most_frequent_baseline'] = score_synset_lists(y_correct_labels, x_lesk_syn_predictions)
    display_dict['lesk_score_raw'] = score_synset_lists(y_correct_labels, x_lesk_syn_predictions)
    display_dict['lesk_score_pos'] = score_synset_lists(y_correct_labels, x_lesk_syn_predictions_pos)
    display_dict['lesk_score_preprocessed'] = score_synset_lists(y_correct_labels, x_lesk_syn_predictions_pre)
    display_dict["lesk_score_preprocessed_pos"] = score_synset_lists(y_correct_labels, x_lesk_syn_predictions_pre_pos)
    
    table = tabulate(display_dict.items(), headers=["Method", "Score"], tablefmt="pretty")
    print(table)
    print(get_semcor_sentences_for_synset(wn.synset('dog.n.01')))
    

    

