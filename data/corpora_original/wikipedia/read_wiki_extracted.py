# from https://github.com/Hironsan/wiki-article-dataset/tree/master

import argparse
import glob
import json
import os
import pandas as pd
import sudachipy
from sudachipy import Dictionary, Tokenizer
import sudachidict_full
#import nltk
from tqdm import tqdm
import re

import spacy


def is_good(sentence, token_limit=50, punct_ratio=0.2, numeral_ratio=0.2, tokenizer=None):
    """Basic surface level requirements for sentence inclusion. requires spacy.
    If a list of tokens is passed, the doc is retrieved.
    If a sentence is passed, it will be tokenized if also the spacy tokenizer is passed, otherwise it will raise exception..."""
    # based on english characters and urls
    # based on having at least 5 tokens [sangawa paper] and ending in punctuation, and ending with a ADJ, VERB, AUX.
    
    if isinstance(sentence, list):
        # retrieve the doc from a list of tokens
        sentence = sentence[0].doc
    elif isinstance(sentence, str):
        sentence = tokenizer(sentence)

    sentence_length = len(sentence) # in tokens!!
    if sentence_length < 5 or sentence_length > token_limit:
        # print('len')
        return False
    
    
    ## TODO: Also consider valid sentences that end with particles.
    
    if (sentence[-1].pos_ != 'PUNCT') or (sentence[-2].pos_ not in ['AUX', 'ADJ', 'VERB']):
    # print('ending')
        return False
    # pattern = r'[a-zA-Z]|https?:\/\/\S+'
    # if re.search(pattern, sentence.text) is not None:
    #     return False
    
    # no more than 20% punctuation or numerals
    #punct_match = re.findall(r'[!\"#$%&\'()*+,-./:;<=>?@[\\\]^_``{|}~…]', sentence.text)
    #punct_count = len(punct_match)
    # we do not consider the last
    punct_count = sum([token.is_punct for token in sentence[0:-1]])
    if punct_count / sentence_length > punct_ratio:
        #print('too much punct',sentence_length, punct_count, punct_count / sentence_length)
        return False
    
    # num_match = re.findall(r'\d', sentence.text)
    # num_count = len(num_match)
    num_count = sum([token.is_digit for token in sentence]) # digit actually means a full token of digits
    if num_count / sentence_length > numeral_ratio:
        #print('too much num',sentence_length, punct_count, punct_count / sentence_length)
        return False
    # no text in other languages
    if re.search(r'.*[A-Za-z].*', sentence.text):
        # print('english')
        return False
    
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F]+'
    if re.search(arabic_pattern, sentence.text):
        # print('arabic')
        return False

    russian_pattern = r'[А-Яа-яЁё]+'
    if re.search(russian_pattern, sentence):
        # print('arabic')
        return False

    return True


# t = Dictionary(dict='full').create(mode=sudachipy.SplitMode.C)
# #t = MeCab.Tagger('-Owakati')
# sent_detector = nltk.RegexpTokenizer(r'(?<!「|『)[\u3000-\u303F\u4E00-\u9FAF\u3040-\u309F\u30A0-\u30FF、]+?[。？！](?!」|』)')


# def tokenize(text):
#     tokens = t.tokenize(text)
#     wakati_tokens = "".join([token.surface() for token in tokens])
#     return wakati_tokens


# def split_text(text):
#     return sent_detector.tokenize(text)


def list_files(dir_path):
    path = os.path.join(dir_path, '*', 'wiki_*')
    files = glob.glob(path, recursive=True)
    return files


def read_jsonl(filepath):
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)


def extract_text(article):
    return article.get('text', '')


def main(args):
    max_chunk_size = int(49140 / 4)  # Maximum chunk size supported by the library

    # nlp = spacy.load('ja_ginza_electra')
    # use only sententicizer
    nlp = spacy.load('ja_ginza_electra', enable='')
    nlp.add_pipe('sentencizer')
    files = list_files(args.extracted_dir)
    with open(args.save_file, 'w') as f:
        with open(args.save_file+'_rejected.csv', 'w') as rejected_f:
            #f.write('sentence\n')
            #rejected_f.write('sentence\n')
            for file in tqdm(files):
                articles = read_jsonl(file)
                for article in articles:
                    text = extract_text(article)
                    text = re.sub(r'[\s\u3000]+', '', text)
                    # need to split into smaller chunks - maybe we lose some sentences...
                    
                    # Split the input text into smaller chunks
                    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                    docs = nlp.pipe(chunks)
                    for doc in docs:
                        for sent in doc.sents:
                            if is_good(sent, token_limit=args.token_limit):
                                f.write(f'{sent}\n')
                            else: rejected_f.write(f'{sent}\n')
                    # for sent in sents:
                    #     sent = sent.strip()
                    #     if len(sent) <= args.char_limit:
                    #         f.write(f'{sent}\n')
                    #sents = [sent.strip() for sent in sents if sent.strip()]
                    #text = '\n'.join(tokenize(sent) for sent in sents)
                    #f.write(f'{text}')
    # now check and clean up 
    rows = []
    with open(args.save_file) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    df = pd.DataFrame(rows, columns=['sentence'])
    series = df['sentence']
    non_strings = series.apply(lambda x: not isinstance(x, str))
    cleaned_series = series[~non_strings].reset_index(drop=True)
    cleaned_series.to_csv(args.save_file, header=False, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Making a dataset.')
    parser.add_argument('--extracted_dir', help='extracted dir by wikiextractor')
    parser.add_argument('--save_file', default='wikipedia.txt', help='filename')
    parser.add_argument('--token_limit', type=int, default=50, help='Max sentence length (in tokens).')
    args = parser.parse_args()
    main(args)
    with open('result.log', 'w') as log:
        log.write('Done')