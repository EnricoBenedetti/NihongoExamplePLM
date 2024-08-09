import re
from collections import defaultdict, deque
from itertools import chain
import sudachipy # https://worksapplications.github.io/sudachi.rs/python/
from sudachipy import Dictionary, Tokenizer
import pickle
import sudachidict_full
import pandas as pd
import os
from datasets import load_dataset
from tqdm import tqdm
import utils
from utils import get_logger
import argparse
import spacy, ginza # pip install -U ginza https://github.com/megagonlabs/ginza/releases/download/latest/ja_ginza_electra-latest-with-model.tar.gz

class InvertedIndex():
    """Class that builds and maintains the inverted index"""
    
    def __init__(self, **config):
        """Config: should contain at least the spacy nlp object ("nlp")"""

        self.doc_number = 0

        if not config:
            # initialize dictionary, tokenizer
            self.nlp = spacy.load("ja_ginza")
            #self.dictionary = Dictionary(dict='full', config_path=None)

        else: 
            for key, value in config.items():
                setattr(self, key, value)
        
        self.tokenizer = self.nlp.tokenizer
        self.inverted_index = defaultdict(list)
        
    def compute_doc_number(self):
        try:
            self.doc_number = max((chain(*self.inverted_index.values()))) + 1
        except ValueError:
            pass
        return self.doc_number


    # def process_basic(self, text, ids=None):
    #     """This does just: normalized form - list of sentence ids
    #     Text is assumed to be an iterable of strings, and id their respective id."""
    #     docs_processed = self.compute_max_doc_id()

    #     if ids == None:
    #         iter = enumerate(text, start=docs_processed)
    #     else: iter = zip(ids, text)

    #     for id, sentence in iter:
    #         tokens = self.tokenizer(sentence)
    #         # for token in set(tokens):  # Use set to ensure each term is indexed only once per document
    #         # this is to store only one of the occurences. (?) idk if correct
    #         for token in set([token.norm_ for token in tokens]):
    #             # add many things
    #             #postings = dict()
    #             #postings['sentence_id'] = self.doc_number
                
    #             # finally add the entry
    #             #self.inverted_index[token.surface()].append({'id':doc_id, 'info': token})
    #             self.inverted_index[token].append(id)
    #     print(f"Processed {self.doc_number - docs_processed} sentences")
    #     return 
    


    def process(self, text, ids=None):
        """This does just: normalized form - list of sentence ids
        Text is assumed to be an iterable of strings, and id their respective id."""

        self.compute_doc_number()

        if ids == None:
            data = data = zip(text, range(len(text)))
        else: data = zip(text, ids)

        # analyse bunstsu with pipe
        pipe = self.nlp.pipe(data, n_process=2, batch_size=1000, as_tuples=True)

        for doc, id in tqdm(pipe):
            #print(id, doc)
            # we fill those: lemma of each token, and lemma of compounds
            bunsetsu_lemmas = []
            lemmas = []
            # extract the phrases
            bun_spans = ginza.bunsetu_phrase_spans(doc)
            for bun_span in bun_spans:
                bunsetsu_lemmas.append(bun_span.lemma_)
            # tokens
            for token in doc:
                lemmas.append(token.lemma_)
        
            index_tokens = set(lemmas + bunsetsu_lemmas)
            # for token in set(tokens):  # Use set to ensure each term is indexed only once per document
            # this is to store only one of the occurences. (?) idk if correct
            for token in index_tokens:
                self.inverted_index[token].append(id)
        print(f"Processed up to id {id}")
        return 

    # def process_bunsetsu(self, text, ids=None):
    #     """This does just: normalized form - list of sentence ids
    #     Text is assumed to be an iterable of strings, and id their respective id."""
    #     docs_processed = self.doc_number

    #     if ids == None:
    #         iter = enumerate(text, start=docs_processed)
    #     else: iter = zip(ids, text)

    #     # analyse bunstsu with pipe
    #     pipe = self.nlp.pipe(text, n_process=2, batch_size=2000)

    #     for id, sentence in tqdm(iter):
    #         # first we analyze bunsetsus to get compound words
    #         # very slow on cpu - see on gpu
    #         doc = next(pipe)
    #         bunsetsu_lemmas = []
    #         for phrase in ginza.bunsetu_phrase_spans(doc):
    #             bunsetsu_lemmas.append(phrase.lemma_)

    #         #tokens = self.tokenizer(sentence)
    #         lemmas = [token.lemma_ for token in doc]
    #         final_tokens = set(lemmas + bunsetsu_lemmas)
    #         # for token in set(tokens):  # Use set to ensure each term is indexed only once per document
    #         # this is to store only one of the occurences. (?) idk if correct
    #         for token in final_tokens:
    #             # add many things
    #             #postings = dict()
    #             #postings['sentence_id'] = self.doc_number
    #             # probably the start and end would be useful
                
    #             # finally add the entry
    #             #self.inverted_index[token.surface()].append({'id':doc_id, 'info': token})
    #             self.inverted_index[token].append(id)
    #         self.doc_number+=1
    #     print(f"Processed {self.doc_number - docs_processed} sentences")
    #     return 

    # def process_experimental(self, text):
    #     """Processes the text and adds it to the index.
    #     Need to think about doing it in a stream? For long files?"""
    #     # default for list of sentences
    #     #print(text)
    #     for doc_id, sentence in enumerate(text):
    #         tokens = self.tokenizer(sentence)
    #         for token in tokens:  # Use set to ensure each term is indexed only once per document
    #             # add many things
    #             postings = dict()
    #             for info in self.postings_additional_info:
    #                     postings[info] = getattr(token, info)()
    #             postings['sentence_id'] = self.doc_number
    #             # finally add the entry
    #             #self.inverted_index[token.surface()].append({'id':doc_id, 'info': token})
    #             self.inverted_index[token.norm_].append(postings)
    #         self.doc_number+=1
    #     print(f"Processed {doc_id} sentences")
    #     return 

    # def process_huggingface(self, examples, idx):
    #     """Given a batch, uses the inverted index to process text, and adds a value containing the doc id to it.
    #     Examples is what is inputted from dataset.map, idx is the idx list from with_indices=True"""
    #     for sentence, id in zip(examples['sentence'], idx):
    #         tokens = self.tokenizer(sentence)
    #         # for token in set(tokens):  # Use set to ensure each term is indexed only once per document
    #         # this is to store only one of the occurences. (?) idk if correct
    #         for token in set([token.norm_ for token in tokens]):
    #             # add many things
    #             #postings = dict()
    #             #postings['sentence_id'] = id
                
    #             # finally add the entry
    #             #self.inverted_index[token.surface()].append({'id':doc_id, 'info': token})
    #             self.inverted_index[token].append(id)
    #         self.doc_number+=1
    #     return {'sentence_id': idx}
    
    # def save_inverted_index(self, dir='./'):
    #     """How to save the file on disk for future use. Also saves the last sentence id for successive addings"""
    #     data = self.inverted_index

    #     file_dir = os.path.realpath(dir)
    #     ## IMPORTANT AVOID: just save in the same data directory
    #     index_file_path = os.path.join(file_dir, "inverted_index.pkl")
    #     doc_file_path = os.path.join(file_dir, "docnum.pkl")

    #     # Store data (serialize)
    #     with open(index_file_path, 'wb') as handle:
    #         pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     with open(doc_file_path, 'wb') as handle:
    #         pickle.dump(self.doc_number, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     print(f"Index saved: {index_file_path}, docs: {doc_file_path}")
    #     return index_file_path, doc_file_path
    
    # def load_inverted_index(self, dir='./'):
    #     """Load the inverted index from pickle format, and the total number of documents"""
    #     self.inverted_index = defaultdict(list)

    #     file_dir = os.path.realpath(dir)

    #     index_file_path = os.path.join(file_dir, "inverted_index.pkl")
    #     doc_file_path = os.path.join(file_dir, "docnum.pkl")

        
    #     # Load data
    #     with open(index_file_path, 'rb') as handle:
    #         self.inverted_index = pickle.load(handle)

    #     with open(doc_file_path, 'rb') as handle:
    #         self.doc_number = pickle.load(handle)

    #     print(f"Index loaded: {index_file_path}, docs: {self.doc_number}")
    #     return 

    def save_inverted_index(self, filename='inverted_index.pkl'):
        """How to save the file on disk for future use. Also saves the last sentence id for successive addings"""
        data = self.inverted_index

        index_file_path = os.path.realpath(filename)
        ## IMPORTANT AVOID: just save in the same data directory
        #index_file_path = os.path.join(file_dir, "inverted_index.pkl")
        #doc_file_path = os.path.join(file_dir, "docnum.pkl")

        # Store data (serialize)
        with open(index_file_path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print(f"Index saved: {index_file_path}")
        return index_file_path
    
    def load_inverted_index(self, filename='inverted_index.pkl'):
        """Load the inverted index from pickle format, and the total number of documents"""
        self.inverted_index = defaultdict(list)

        index_file_path = os.path.realpath(filename)

        #index_file_path = os.path.join(file_dir, "inverted_index.pkl")
        #doc_file_path = os.path.join(file_dir, "docnum.pkl")

        
        # Load data
        with open(index_file_path, 'rb') as handle:
            self.inverted_index = pickle.load(handle)


        print(f"Index loaded: {index_file_path}")
        self.compute_doc_number()
        return 
    
    def search_term(self, query_string="サンプル", override=False):
        """My idea for improving retrieval is, having indexed with bunsetsu also, to:
        1. Tokenize the text passed to the function (should be one word...)
            2. If the result is only one token, then retrieve that token's lemma from the index
            Otherwise, query on the compounding of the token's lemmas.
        A thing left to check is weird compounds.. e.g. verbs maybe get split differently...
        X. Else just do a full text search elsewhere?
        
        input: 
        query: the string
        override: if true, the passed query will be used directly on the index.
        
        output:
        results: list of id
        query: the actual query used (debug)
        query_doc: the tokenized query (spacy)"""

        results = []
        
        
        if override:
            query = query_string
            query_doc = self.tokenizer(query_string)
        else:
            query_doc = self.tokenizer(query_string)
            query = "".join([token.lemma_ for token in query_doc])
                
        results = self.inverted_index.get(query, [])
        return results, query, query_doc


# the idea is processing .csv files adding the sentences to the inverted index, and saving a copy of the dataset with the same index
# should also consider to load from file the already made thing
if __name__ == "__main__":
    logger = get_logger()
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process a list of files")

    # Add an argument to accept a list of files (input from the command line)
    parser.add_argument("--files", nargs="+", help="List of files to process. Should be .csv and contain the 'sentence' column")
    parser.add_argument("--inverted_index_in_file", default=None, help="optional inverted index file")
    parser.add_argument("--out_file", default='../../data/indexed_data.pkl', help="where to save the index")
    #parser.add_argument('--process_mode', default='bunsetsu', help='how to get the tokens from the sentences. default uses bunsetsu and normalized forms. basic uses only normalized forms')
    

    # HOW TO RUN:
    # nohup python inverted_index.py --files /data/enrico_benedetti/nihongoexample/data/corpus_all/jp_sentences/corpus.csv --out_file /data/enrico_benedetti/nihongoexample/data/corpus_all/jp_sentences/inverted_index.pkl & 

    # Parse the command line arguments
    args = parser.parse_args()
    logger.debug(args)

    out_file = os.path.realpath(args.out_file)
    out_dir, _ = os.path.split(out_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    index = InvertedIndex()

    #default
    #process_function = index.process_basic
    #if args.process_mode == 'bunsetsu':
    #    process_function = index.process_bunsetsu
    
    inverted_index_in_file = args.inverted_index_in_file
    if inverted_index_in_file:
        inverted_index_in_file = os.path.realpath(inverted_index_in_file)
        logger.info(f'Trying to load the inverted index from {inverted_index_in_file}')
        try:
            index.load_inverted_index(inverted_index_in_file)
        except FileNotFoundError as e:
            logger.error(e)

    else: logger.info(f'Creating new inverted index')

    # process all files passed
    for file in tqdm(args.files):
        try:
            logger.info(f'processing {file}')

            df_file_name = os.path.realpath(file)
            dir_name, file_name = os.path.split(df_file_name)

            # Split the file name into name and extension
            name, ext = os.path.splitext(file_name)

            # Append '_new' to the name and keep the extension
            new_file_name = f"{name}_index{ext}"

            # Reassemble the full path with the modified file name
            new_df_file_name = os.path.join(out_dir, new_file_name)
            
            df = pd.read_csv(df_file_name)
            # reset index of dataframe and add the currently processed documents
            df_reset = df.reset_index(drop=True)
            index.compute_doc_number()
            df_reset.index += index.doc_number
            df_reset.to_csv(new_df_file_name)
            logger.info(f'new dataset saved in {new_df_file_name}')

            # process sentences
            logger.debug(f'processing sentences...')
            index.process(df['sentence'])

        except Exception as e:
            logger.error(e)
            continue

    
    
    index.save_inverted_index(filename=out_file)

    logger.info(f'saving inverted index in {out_file}')
    logger.info('finished')