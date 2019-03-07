from __future__ import unicode_literals, print_function
import spacy
import jieba
import random
import ipdb
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import csv
import time
from tqdm import tqdm
import codecs
spacy.__version__


def init_model(path):
    # the model file path 
    return spacy.load(path)

def ner(model, msg):
    doc = model(msg)
    
    # [('远光灯', 'high_beam')]
    return [(ent.text, ent.label_) for ent in doc.ents]

if __name__ == "__main__":
    pass