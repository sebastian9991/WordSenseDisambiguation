from loader import load_instances, load_key
import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import stopwords
from tabulate import tabulate

'''
Sense indecates as #1 in the synset according to WordNet
'''
def most_frequent_baseline(sense_key): 
    return wn.synset_from_sense_key(sense_key=sense_key) 

'''
NLTK implementation of Lesk's WSD
'''
def lesk_wsd(context, lemma, pos = None):
    return lesk(context_sentence=context, ambiguous_word= lemma
                , pos=pos)
    
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
def download_and_get_nltk_tools():
    #Create tools
    stop_words = stopwords.words('english')
    return stop_words

'''
Preprocess the context sentence, namely remove the stop words
'''
def preprocess_context_sentence(context_list):
    stop_words = download_and_get_nltk_tools()
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
Yarowasky's bootstrap method refered from 
Speech & Language Processing, Jurafsky, D.2nd (Page 650)
'''
def yarowsky_bootstrap():
    return 0
    

'''
used from @jcheung start code
Main method
'''
if __name__ == '__main__':
    nltk.download('stopwords')
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

    display_dict = {
        "lesk_score_raw": 0,
        "lesk_score_pos" : 0,
        "lesk_score_preprocessed": 0, 
        "lesk_score_preprocessed_pos": 0
    }
    y_most_frequent_baseline = []
    x_lesk_syn_predictions = []
    x_lesk_syn_predictions_pos = []
    x_lesk_syn_predictions_pre = []
    x_lesk_syn_predictions_pre_pos = []
    for item, key in zip(dev_instances.items(), dev_key):
        syn_key = dev_key[key][0]
        y_most_frequent_baseline.append(most_frequent_baseline(syn_key))

        ##Lesk() requires strings not bytes, throws error when input is b''
        wsd_instance = item[1]
        context = list(map(lambda x: x.decode("utf-8"), wsd_instance.context))
        lemma = wsd_instance.lemma.decode("utf-8")
        pos = wsd_instance.pos.decode("utf-8")
        x_lesk_syn_predictions.append(lesk_wsd(context, lemma))
        x_lesk_syn_predictions_pos.append(lesk_wsd(context, lemma, pos = pos_tag_mapper(pos)))
        
        ##After preprocessing
        context_pre = preprocess_context_sentence(context_list=context)
        x_lesk_syn_predictions_pre.append(lesk_wsd(context_pre, lemma))
        x_lesk_syn_predictions_pre_pos.append(lesk_wsd(context=context_pre, lemma = lemma, pos=pos_tag_mapper(pos)))




    
    display_dict['lesk_score_raw'] = score_synset_lists(y_most_frequent_baseline, x_lesk_syn_predictions)
    display_dict['lesk_score_pos'] = score_synset_lists(y_most_frequent_baseline, x_lesk_syn_predictions_pos)
    display_dict['lesk_score_preprocessed'] = score_synset_lists(y_most_frequent_baseline, x_lesk_syn_predictions_pre)
    display_dict["lesk_score_preprocessed_pos"] = score_synset_lists(y_most_frequent_baseline, x_lesk_syn_predictions_pre_pos)
    
    table = tabulate(display_dict.items(), headers=["Method", "Score"], tablefmt="pretty")
    print(table)

    

