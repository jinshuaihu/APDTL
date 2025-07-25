"""Vocabulary wrapper"""
import os
import json
import nltk
import argparse
from collections import Counter


annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
}


class Vocabulary(object):
    """
        Simple vocabulary wrapper.
    """

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, caption_file, threshold):
    """
        Build a simple vocabulary wrapper.
    """
    counter = Counter()
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        captions = from_txt(full_path)
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower())
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def build_vocab_conceptual(base_dir, threshold):
    """
        Build a simple vocabulary wrapper.
    """
    counter = Counter()
    files = ['Train_GCC-training.tsv', ]
    for filename in files:
        file_path = os.path.join(base_dir, filename)
        df = open_tsv(file_path)
        for i, row in enumerate(df.iterrows()):
            caption = row[1]['caption']
            tokens = nltk.tokenize.word_tokenize(
                caption.lower())
            counter.update(tokens)
            if i % 10000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(df)))
    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<mask>') 
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def open_tsv(fname):
    import pandas as pd
    print("Opening %s Data File..." % fname)
    df = pd.read_csv(fname, sep='\t', names=["caption", "url"], usecols=range(0, 2))
    print("Processing", len(df), " Images:")
    return df


def main(data_path, data_name):
    print('1.', data_path)
    print('2.', data_name)
    vocab = build_vocab(data_path, data_name, caption_file=annotations, threshold=4)
    serialize_vocab(vocab, './vocab/%s_vocab.json' % data_name)
    print("Saved vocabulary file to ", './vocab/%s_vocab.json' % data_name)
