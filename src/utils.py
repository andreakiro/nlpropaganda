from typing import List

from allennlp.data.tokenizers.token_class import Token

labels = {
    'Not_Propaganda': 0,
    'Loaded_Language': 1,
    'Name_Calling,Labeling': 2,
    'Repetition': 3,
    'Exaggeration,Minimisation': 4,
    'Doubt': 5,
    'Appeal_to_fear-prejudice': 6,
    'Flag-Waving': 7,
    'Causal_Oversimplification': 8,
    'Slogans': 9,
    'Appeal_to_Authority': 10,
    'Black-and-White_Fallacy': 11,
    'Thought-terminating_Cliches': 12,
    'Whataboutism,Straw_Men,Red_Herring': 13,
    'Bandwagon,Reductio_ad_hitlerum': 14,
}

def label_to_int(label):
    return labels[label]

def int_to_label(i):
    for key, value in labels.items():
        if value == i:
            return key
    raise ValueError(f'Key for value {i} doesnt exist.')

def get_not_propaganda():
    return labels['Not_Propaganda']

def filter_function(span_tokens: List[Token]):
    return not all([t.pos_ == "DET" or t.pos_ == "PUNCT" or t.pos_ == "EOL" or t.pos_ == "SPACE" for t in span_tokens])